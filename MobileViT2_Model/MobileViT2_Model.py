import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime

# --- 하이퍼파라미터 설정 (여기서 한 번에 관리하세요) ---
IMG_SIZE = 224  # 256에서 224로 하향 조정 (메모리 확보)
BATCH_SIZE = 16  # 4에서 8로 살짝 올려보되, 또 터지면 4로 낮추세요
EPOCHS = 5
LEARNING_RATE = 0.0001
DATA_PATH = './data/train/'

def conv_block(x, filters, kernel_size=3, strides=1, activation=tf.nn.swish):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def mv2_block(x, expansion_factor, out_channels, strides):
    in_channels = x.shape[-1]
    hidden_dim = int(in_channels * expansion_factor)
    
    res = conv_block(x, hidden_dim, kernel_size=1)
    res = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", use_bias=False)(res)
    res = layers.BatchNormalization()(res)
    res = layers.Activation(tf.nn.swish)(res)
    res = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(res)
    res = layers.BatchNormalization()(res)
    
    if strides == 1 and in_channels == out_channels:
        return layers.Add()([x, res])
    return res

def mobilevit_block(x, num_transformer_blocks, projection_dim, patch_size=2):
    local_features = conv_block(x, projection_dim, kernel_size=3)
    local_features = conv_block(local_features, projection_dim, kernel_size=1, activation=None)
    
    _, h, w, c = local_features.shape
    num_patches = (h * w) // (patch_size ** 2)
    # Patch-wise reshaping
    global_features = layers.Reshape((patch_size ** 2, num_patches, projection_dim))(local_features)
    
    for _ in range(num_transformer_blocks):
        # Attention
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=projection_dim)(global_features, global_features)
        attn_out = layers.Add()([global_features, attn_out])
        attn_out = layers.LayerNormalization()(attn_out)
        
        # Feed Forward
        ffn_out = layers.Dense(projection_dim * 2, activation=tf.nn.swish)(attn_out)
        ffn_out = layers.Dense(projection_dim)(ffn_out)
        global_features = layers.Add()([attn_out, ffn_out])
        global_features = layers.LayerNormalization()(global_features)
        
    global_features = layers.Reshape((h, w, projection_dim))(global_features)
    
    # x(Skip connection)와 global_features의 채널 수를 맞춰서 합쳐야 합니다.
    # 만약 x의 채널이 다르면 1x1 conv로 맞춰줍니다.
    if x.shape[-1] != projection_dim:
        x_skip = conv_block(x, projection_dim, kernel_size=1, activation=None)
    else:
        x_skip = x
        
    combined = layers.Concatenate()([x_skip, global_features])
    out = conv_block(combined, projection_dim, kernel_size=3)
    return out

def build_mobilevit_v2(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
    x = conv_block(inputs, 32, strides=2)
    x = mv2_block(x, expansion_factor=2, out_channels=64, strides=1)
    
    x = mv2_block(x, expansion_factor=2, out_channels=128, strides=2)
    x = mobilevit_block(x, num_transformer_blocks=2, projection_dim=128)
    
    x = mv2_block(x, expansion_factor=2, out_channels=256, strides=2)
    x = mobilevit_block(x, num_transformer_blocks=3, projection_dim=256)
    
    x = conv_block(x, 512, kernel_size=1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    
    return Model(inputs, outputs, name="MobileViT_v2")

if __name__ == "__main__":
    # GPU 메모리 점진적 할당 설정 (OOM 방지)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU 메모리 유연 할당 설정 완료")
        except RuntimeError as e:
            print(e)

    # 1. 모델 빌드 (IMG_SIZE 변수 사용)
    model = build_mobilevit_v2(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 2. 데이터 준비
    datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
    
    print(f"📂 데이터를 불러오는 중: {DATA_PATH}")
    train_gen = datagen.flow_from_directory(
        DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # 3. 학습 시작
    print(f"🚀 학습 시작 (이미지 크기: {IMG_SIZE}, 배치 크기: {BATCH_SIZE})...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    
    # 4. 결과 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f'mobilevit_model_{timestamp}.h5'

    model.save(model_filename)
    print(f"✅ 모델이 {model_filename}으로 저장되었습니다!")