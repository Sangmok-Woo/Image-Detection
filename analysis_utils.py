import os
import sys
import datetime
import keras

sys.modules['tensorflow.keras'] = keras 
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import requests
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# 1. Grad-CAM 히트맵 생성 및 시각화 엔진
# ---------------------------------------------------------------------------

def _get_last_conv_layer(model):
    """
    모델에서 GAP 직전 마지막 Conv2D 레이어를 자동으로 찾습니다.
    MobileViT v2 아키텍처 기준: conv_block(512) 레이어가 타겟입니다.
    """
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
    if last_conv is None:
        raise ValueError("모델에서 Conv2D 레이어를 찾을 수 없습니다.")
    return last_conv.name


def get_gradcam_heatmap(img_array: np.ndarray, model) -> np.ndarray:
    """
    Grad-CAM 알고리즘으로 히트맵 배열을 생성합니다.
    """
    last_conv_layer_name = _get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def overlay_heatmap_on_image(
    original_img_bytes: bytes,
    heatmap: np.ndarray,
    alpha: float = 0.45
) -> Image.Image:
    """
    원본 이미지 위에 Grad-CAM 히트맵을 컬러맵으로 오버레이합니다.
    """
    original = Image.open(io.BytesIO(original_img_bytes)).convert('RGB')
    original_resized = original.resize((224, 224))

    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((224, 224), Image.BILINEAR)
    ) / 255.0

    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    heatmap_colored = np.uint8(heatmap_colored * 255)

    original_arr = np.array(original_resized, dtype=np.float32)
    heatmap_arr = np.array(heatmap_colored, dtype=np.float32)
    blended = (1 - alpha) * original_arr + alpha * heatmap_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={'width_ratios': [6, 0.35]})

    axes[0].imshow(blended)
    axes[0].axis('off')
    axes[0].set_title('Grad-CAM Heatmap', fontsize=14, pad=12)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(
        axes[1], cmap=cm.jet, norm=norm, orientation='vertical'
    )
    cb.set_label('Activation', fontsize=10)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Low', 'Mid', 'High'])
    cb.ax.tick_params(labelsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def generate_gradcam_overlay(
    image_bytes: bytes,
    model
) -> Image.Image:
    """
    app.py에서 호출하는 Grad-CAM 내부 진입점
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = get_gradcam_heatmap(img_array, model)
    result_img = overlay_heatmap_on_image(image_bytes, heatmap)

    return result_img


# ---------------------------------------------------------------------------
# 2. VLM / LLM 분석 파이프라인 (터미널 로그 관측 시스템 내장)
# ---------------------------------------------------------------------------

def analyze_prediction(prob: float) -> dict:
    is_real = prob >= 0.5
    verdict = "REAL" if is_real else "AI Generated"

    confidence_score = prob if is_real else (1.0 - prob)
    confidence_pct = round(confidence_score * 100, 2)

    if confidence_score >= 0.90:
        confidence_level = "매우 높음"
        confidence_desc = "모델이 강한 확신을 가지고 판별했습니다."
    elif confidence_score >= 0.75:
        confidence_level = "높음"
        confidence_desc = "모델이 비교적 명확한 근거로 판별했습니다."
    elif confidence_score >= 0.60:
        confidence_level = "보통"
        confidence_desc = "일부 특징이 경계선 근처에 위치하여 불확실성이 존재합니다."
    else:
        confidence_level = "낮음"
        confidence_desc = "판별 경계값(0.5)에 근접합니다. 두 클래스의 특성을 동시에 보유할 수 있습니다."

    margin_from_boundary = round(abs(prob - 0.5), 4)
    boundary_proximity = "경계선 근접" if margin_from_boundary < 0.15 else "경계선과 충분한 거리"

    texture_analysis = {
        "overall_naturalness": round(confidence_score, 3),
        "edge_sharpness": "자연적" if is_real else "인공적 과선명 또는 과부드러움 가능성",
        "noise_distribution": "자연적 센서 노이즈 패턴" if is_real else "균일하거나 비자연적인 노이즈 분포 가능성",
        "lighting_coherence": "일관된 광원 처리" if is_real else "광원 불일치 또는 과도한 HDR 처리 가능성",
        "frequency_pattern": "정상 주파수 분포" if is_real else "고주파 성분의 비자연적 분포 가능성",
    }

    return {
        "verdict": verdict,
        "raw_probability": round(float(prob), 6),
        "is_real": is_real,
        "confidence_score": confidence_score,
        "confidence_pct": confidence_pct,
        "confidence_level": confidence_level,
        "confidence_description": confidence_desc,
        "margin_from_boundary": margin_from_boundary,
        "boundary_proximity": boundary_proximity,
        "texture_analysis": texture_analysis,
        "model_name": "MobileViT v2",
        "decision_threshold": 0.5,
    }


def build_llm_prompt(
    analysis: dict,
    image_base64: str | None = None,
    heatmap_base64: str | None = None,
    image_media_type: str = "image/jpeg",
):
    system_context = """당신은 컴퓨터 비전 및 생성형 AI 탐지 전문가입니다.

사용자가 업로드한 원본 이미지와 MobileViT v2의 판정 결과,
그리고 제공되는 경우 Grad-CAM 히트맵을 함께 분석하여
모델이 왜 해당 결론을 내렸는지 설명해주세요.

중요한 원칙:
1. 원본 이미지를 실제로 관찰한 뒤 설명합니다.
2. 이미지에서 실제로 확인할 수 있는 시각적 특징을 구체적으로 언급합니다.
3. 관찰한 특징과 MobileViT v2의 판정 결과가 어떻게 연결되는지 설명합니다.
4. 이미지에서 확인할 수 없는 내용은 추측하거나 단정하지 않습니다.
5. Grad-CAM 히트맵이 제공된 경우에만 모델이 주목한 영역을 언급합니다.
6. 원본 이미지와 Grad-CAM에서 확인한 내용을 서로 구분해서 설명합니다.
7. 모델의 신뢰도가 낮거나 결정 경계(0.5)에 가까우면 확정적으로 표현하지 않습니다.
8. 단순히 AI/REAL 판정 결과를 반복하지 말고 이미지와 판정 결과의 관계를 설명합니다.
9. 3~5문장으로 간결하게 한국어로 답변합니다."""

    analysis_text = f"""## MobileViT v2 추론 결과

- 판정: {analysis['verdict']}
- 확률값(sigmoid): {analysis['raw_probability']} (0=AI생성, 1=실제)
- 신뢰도: {analysis['confidence_pct']}% ({analysis['confidence_level']})
- {analysis['confidence_description']}
- 결정 경계(0.5)로부터 거리: {round(analysis['margin_from_boundary'] * 100, 1)}% ({analysis['boundary_proximity']})

### 기존 분석 데이터
- 엣지 처리: {analysis['texture_analysis']['edge_sharpness']}
- 노이즈 분포: {analysis['texture_analysis']['noise_distribution']}
- 광원 일관성: {analysis['texture_analysis']['lighting_coherence']}
- 주파수 패턴: {analysis['texture_analysis']['frequency_pattern']}

첫 번째 이미지는 원본 입력 이미지입니다.
두 번째 이미지가 제공된다면 그것은 MobileViT v2의 Grad-CAM 시각화입니다.

원본 이미지에서 실제로 관찰되는 특징을 먼저 설명하고,
그 특징이 MobileViT v2의 판정 결과와 어떻게 연결되는지 설명해주세요.
Grad-CAM이 제공된 경우에는 모델이 주목한 영역을 함께 설명해주세요."""

    user_content = []

    # 1. 원본 이미지
    if image_base64:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_base64
            }
        })

    # 2. Grad-CAM 이미지
    if heatmap_base64:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": heatmap_base64
            }
        })

    user_content.append({
        "type": "text",
        "text": analysis_text
    })

    return [{"role": "user", "content": user_content}], system_context


def call_llm(messages: list, system_context: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            "⚠️ [Claude API] ANTHROPIC_API_KEY 환경 변수가 식별되지 않아 "
            "기본 내장 문구를 출력합니다."
        )
        return _fallback_explanation(messages)

    api_key = api_key.strip()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n==================================================================")
    print(f"[{current_time}] 🚀 [Claude API] 분석 요청을 송신합니다...")
    print("  - 모델: claude-sonnet-4-6")
    print("  - 원본 이미지 + Grad-CAM 전달")
    print("==================================================================")

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 600,
                "system": system_context,
                "messages": messages,
            },
            timeout=30
        )

        resp_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"[{resp_time}] ✅ [Claude API] "
            f"서버 응답 수신 - HTTP {response.status_code}"
        )

        response.raise_for_status()

        data = response.json()

        return "\n".join(
            b["text"]
            for b in data.get("content", [])
            if b.get("type") == "text"
        ).strip()

    except requests.exceptions.Timeout:
        print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            "❌ [Claude API Error] 요청 시간이 초과되었습니다."
        )
        return "⚠️ LLM 분석 요청 시간이 초과되었습니다."

    except Exception as e:
        print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"❌ [Claude API Error] {str(e)}"
        )

        if 'response' in locals() and response is not None:
            print(f"Raw Payload: {response.text}")

        return f"⚠️ 분석 중 오류: {str(e)}"


def get_vlm_explanation(
    prob: float,
    result_word: str,
    image_base64: str | None = None,
    heatmap_base64: str | None = None,
    image_media_type: str = "image/jpeg",
) -> str:

    analysis = analyze_prediction(prob)

    messages, system_context = build_llm_prompt(
        analysis,
        image_base64=image_base64,
        heatmap_base64=heatmap_base64,
        image_media_type=image_media_type,
    )

    return call_llm(messages, system_context)