# -*- coding: utf-8 -*-
"""RetinaAI: Retinal image analysis and clinical decision support."""

import io
import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="RetinaAI | Retinal Analysis & Clinical Decision Support",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""<style>
.stApp {background:#f4f7fb;color:#16263f}
.block-container {padding-top:1.5rem;max-width:1350px}
.hero {background:linear-gradient(115deg,#102b4e,#145e82);color:white;padding:26px 30px;border-radius:18px;margin-bottom:20px}
.hero h1 {color:white;margin:0;font-size:2rem}
.hero p {color:#d8eaf3;margin:8px 0 0}
.panel {background:white;border:1px solid #dfe7ef;padding:18px 20px;border-radius:14px;margin-bottom:14px}
.small {font-size:0.88rem;color:#52657d}
</style>""", unsafe_allow_html=True)

st.markdown('''
<div class="hero">
<h1>👁 RetinaAI | Retinal Screening & Clinical Decision Support</h1>
<p>Fundus image analysis • Retinal disease assessment • AI-assisted clinical interpretation</p>
</div>
''', unsafe_allow_html=True)

st.caption(
    "AI-assisted retinal image analysis • Final assessment by ophthalmologist"
)

with st.sidebar:
    st.title("RetinaAI")
    st.caption("Retinal Clinical Decision Support | Version 3.0")

    page = st.radio(
        "Navigate",
        ["Retinal screening", "Clinical workflow", "About this concept"]
    )

    st.divider()

    st.info(
        "AI-assisted retinal image analysis and clinical decision support for ophthalmologists."
    )

if page == "Clinical workflow":

    st.header("RetinaAI Clinical Workflow")

    st.markdown("""
    **1.** Upload retinal fundus photograph

    **2.** Review image quality and retinal structures

    **3.** AI-assisted diabetic retinopathy screening

    **4.** Diabetic retinopathy severity grading

    **5.** Review model classification scores

    **6.** Final clinical assessment by ophthalmologist

    **7.** Generate retinal screening report
    """)

    st.stop()

if page == "About this concept":

    st.header("About RetinaAI")

    st.write(
        "RetinaAI is an AI-assisted retinal image analysis and clinical "
        "decision support application designed to support ophthalmologists "
        "in reviewing fundus photographs and assessing retinal disease."
    )

    st.write(
        "The current diabetic retinopathy classifier uses a fine-tuned "
        "MobileNetV3-Large deep learning architecture."
    )

    st.subheader("Model development and evaluation")

    st.write("Training dataset: 2,706 images")
    st.write("Validation dataset: 549 images")
    st.write("Test dataset: 551 images")
    st.write("Four-class test accuracy: 84.39%")
    st.write("DR detection sensitivity: 98.93%")
    st.write("DR detection specificity: 98.15%")
    st.write("Advanced DR grading recall: 68.92%")

    st.caption(
        "Results are from internal development testing. "
        "Patient-level independence and external clinical validation "
        "have not been established."
    )

    st.write(
        "Additional retinal disease classification and automated "
        "lesion identification are under development."
    )

    st.stop()


# SECTION 1: CASE INFORMATION

st.subheader("1 · Case information")

c1, c2, c3, c4 = st.columns(4)

with c1:
    case_id = st.text_input(
        "Case ID",
        "CASE-001"
    )

with c2:
    eye = st.selectbox(
        "Eye",
        ["Right (OD)", "Left (OS)"]
    )

with c3:
    age_band = st.selectbox(
        "Age group",
        ["Not specified", "Below 40", "40–59", "60–79", "80+"]
    )

with c4:
    diabetes = st.selectbox(
        "Diabetes history",
        ["Not specified", "Reported", "Not reported"]
    )


# SECTION 2: FUNDUS PHOTOGRAPH

st.subheader("2 · Fundus photograph")

source = st.radio(
    "Image source",
    ["Use sample illustration", "Upload fundus photograph"],
    horizontal=True
)

if source == "Upload fundus photograph":

    uploaded = st.file_uploader(
        "Choose retinal fundus photograph (JPG or PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        try:
            image = Image.open(uploaded).convert("RGB")

            if image.width * image.height > 24_000_000:

                st.error(
                    "Image is too large. Please use an image "
                    "smaller than 24 megapixels."
                )

                st.stop()

        except Exception:

            st.error("Unable to open this retinal image.")
            st.stop()

    else:

        st.info(
            "Please upload a retinal photograph to begin analysis."
        )

        st.stop()

else:

    image = Image.open(
        ROOT / "sample_retina.png"
    ).convert("RGB")


left, right = st.columns(2, gap="large")

with left:

    st.markdown("**Original retinal photograph**")

    st.image(
        image,
        use_container_width=True
    )

with right:

    st.markdown("**Retinal image analysis**")

    if source == "Use sample illustration":

        st.image(
            str(ROOT / "sample_retina_annotated.png"),
            use_container_width=True
        )

        st.caption("Illustrative retinal feature markers.")

    else:

        st.info(
            "The uploaded retinal photograph will be assessed "
            "using the diabetic retinopathy classification model."
        )

        st.caption(
            "Automated lesion localization is currently unavailable."
        )


# SECTION 3: DIABETIC RETINOPATHY ASSESSMENT

st.subheader("3 · Diabetic Retinopathy Assessment")

st.caption(
    "MobileNetV3-Large | Four-grade diabetic retinopathy classification"
)

GRADE_NAMES = [
    "No apparent DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR or PDR"
]


@st.cache_resource(
    show_spinner="Loading RetinaAI MobileNetV3 model..."
)
def load_retinal_model():

    import torch
    from torch import nn
    from torchvision import models, transforms

    model = models.mobilenet_v3_large(
        weights=None
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        4
    )

    checkpoint_path = ROOT / "retina_dr_model_v3.pth"

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "retina_dr_model_v3.pth was not found "
            "in the project folder."
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        ),
        strict=True
    )

    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return preprocess, model


image_fingerprint = hashlib.sha256(
    image.tobytes()
).hexdigest()

analysis_key = (
    f"mobilenet_v3:{source}:{image.size}:{image_fingerprint}"
)

if st.session_state.get("retinal_image_fingerprint") != analysis_key:

    st.session_state.pop("retinal_prediction", None)

    st.session_state["retinal_image_fingerprint"] = analysis_key


if st.button(
    "Analyze retinal photograph",
    type="primary",
    disabled=(source == "Use sample illustration")
):

    try:

        import torch

        with st.spinner(
            "Analyzing retinal photograph on CPU..."
        ):

            preprocess, retinal_model = load_retinal_model()

            tensor = preprocess(image).unsqueeze(0)

            with torch.inference_mode():

                logits = retinal_model(tensor)

                scores = torch.softmax(
                    logits,
                    dim=1
                )[0].cpu().tolist()

            if len(scores) != 4:

                raise ValueError(
                    "The model did not return four class scores."
                )

            grade = int(
                max(range(4), key=lambda idx: scores[idx])
            )

            st.session_state["retinal_prediction"] = {
                "grade": grade,
                "scores": scores
            }

    except Exception as exc:

        st.session_state.pop(
            "retinal_prediction",
            None
        )

        st.error(
            f"Unable to run retinal classifier: {exc}"
        )


prediction = st.session_state.get("retinal_prediction")

if prediction is None:

    st.info(
        "Upload a retinal photograph and click "
        "Analyze retinal photograph."
    )

    finding_text = "Retinal analysis not performed."

else:

    grade = prediction["grade"]

    if grade == 0:

        binary = "No DR detected"

        severity = "Grade 0 — No apparent DR"

    else:

        binary = "DR suspected"

        severity = f"Grade {grade} — {GRADE_NAMES[grade]}"

    metric_left, metric_right = st.columns(2)

    with metric_left:

        st.metric(
            "DR screening result",
            binary
        )

    with metric_right:

        st.metric(
            "Predicted DR severity",
            severity
        )

    if grade == 0:

        st.info(
            "The model predicts no apparent diabetic retinopathy. "
            "This does not exclude other retinal diseases."
        )

    elif grade == 1:

        st.warning(
            "The model predicts mild non-proliferative "
            "diabetic retinopathy."
        )

    elif grade == 2:

        st.warning(
            "The model predicts moderate non-proliferative "
            "diabetic retinopathy. Ophthalmologist review is required."
        )

    elif grade == 3:

        st.error(
            "The model predicts advanced diabetic retinopathy "
            "(severe NPDR or PDR). Prompt ophthalmologist "
            "assessment is recommended."
        )

    st.markdown("**Model classification scores**")

    for idx, score in enumerate(prediction["scores"]):

        st.write(
            f"Grade {idx} · {GRADE_NAMES[idx]}: {score:.1%}"
        )

        st.progress(float(score))

    st.caption(
        "Classification scores are not clinically calibrated "
        "probabilities. Image quality and other retinal diseases "
        "are not automatically assessed."
    )

    finding_text = (
        f"AI-assisted DR assessment: {binary}; "
        f"predicted severity: {severity}. "
        f"Predicted class score: "
        f"{prediction['scores'][grade]:.1%}. "
        "Model: RetinaAI MobileNetV3 Version 3. "
        "Other retinal diseases are not excluded. "
        "Final ophthalmologist assessment pending."
    )


if source == "Use sample illustration":

    st.caption(
        "The sample image is an illustration. "
        "Upload a retinal photograph for actual image assessment."
    )

# SECTION 4: AI-ASSISTED CLINICAL EXPLANATION

st.subheader("4 · AI-assisted Clinical Explanation")

st.caption(
    "Qwen2-VL-2B-Instruct | Clinical explanation of DR classification"
)


@st.cache_resource(
    show_spinner="Loading Qwen model on CPU..."
)
def load_qwen_model():

    import torch
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor
    )

    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=True
    )

    model.to("cpu")
    model.eval()

    return processor, model


explanation_key = f"qwen_explanation_{analysis_key}"

explanation = "AI clinical explanation not generated."

if prediction is None:

    st.info(
        "Complete the diabetic retinopathy analysis first."
    )

else:

    if st.button("Generate AI Clinical Explanation"):

        try:

            import torch

            with st.spinner(
                "Qwen is generating the clinical explanation..."
            ):

                processor, qwen_model = load_qwen_model()

                prompt = f"""
You are a clinical decision support assistant
supporting an ophthalmologist.

The MobileNetV3 retinal classification model
has produced the following results:

DR screening result: {binary}
Predicted severity: {severity}
Predicted class score:
{prediction['scores'][grade]:.1%}

Diabetes history: {diabetes}
Age group: {age_band}
Eye: {eye}

Provide a concise clinical explanation using
the following headings:

1. AI Classification Summary
2. Clinical Significance
3. Recommended Clinical Assessment
4. Limitations

Explain the supplied classification result.

Do not claim to have examined the retinal image.
Do not invent hemorrhages, microaneurysms,
exudates or neovascularization.
Do not diagnose other retinal diseases.
The class score is not a calibrated probability.
Final diagnosis requires ophthalmologist review.

Keep the explanation under 180 words.
"""

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt"
                )

                inputs = inputs.to("cpu")

                with torch.inference_mode():

                    generated_ids = qwen_model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=False
                    )

                generated_ids = generated_ids[
                    :, inputs.input_ids.shape[1]:
                ]

                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                st.session_state[explanation_key] = {
                    "text": response,
                    "context": (case_id, eye, age_band, diabetes)
                }

        except Exception as exc:

            st.error(f"Qwen explanation error: {exc}")

    saved = st.session_state.get(explanation_key)

    if saved and saved["context"] == (
        case_id, eye, age_band, diabetes
    ):

        explanation = saved["text"]

        st.markdown(explanation)

        st.caption(
            "AI-generated explanation of classification results. "
            "Not an independent retinal image diagnosis."
        )


# SECTION 5: OPHTHALMOLOGIST REVIEW

st.subheader("5 · Ophthalmologist Review & Clinical Report")

reviewer = st.selectbox(
    "Clinical review status",
    [
        "Not reviewed",
        "Pending ophthalmologist review",
        "Reviewed by ophthalmologist"
    ],
    key=f"reviewer_{analysis_key}_{case_id}_{eye}"
)

notes = st.text_area(
    "Ophthalmologist clinical assessment",
    placeholder=(
        "Enter clinical findings, interpretation, "
        "and final assessment..."
    ),
    key=f"notes_{analysis_key}_{case_id}_{eye}"
)


# PDF REPORT GENERATION

def build_pdf():

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=(595, 842),
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    safe = lambda s: (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story = [

        Paragraph(
            "RetinaAI — AI-Assisted Retinal Assessment Report",
            styles["Title"]
        ),

        Spacer(1, 12),

        Paragraph(
            "Retinal Image Analysis and Ophthalmologist Assessment",
            styles["Heading2"]
        ),

        Spacer(1, 12)
    ]

    for key, value in [

        ("Case ID", case_id),

        ("Eye", eye),

        ("Age group", age_band),

        ("Diabetes history", diabetes),

        ("Image source", source),

        ("AI model", "MobileNetV3-Large Version 3"),

        ("AI retinal assessment", finding_text),

        ("AI visual retinal review", explanation),

        ("Clinical review status", reviewer),

        ("Final ophthalmologist assessment", notes or "Pending"),

        (
            "Clinical limitation",
            "Research prototype. External clinical validation "
            "is pending. Not a standalone diagnostic system."
        )

    ]:

        story.append(
            Paragraph(
                f"<b>{safe(key)}:</b> {safe(value)}",
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 9))

    doc.build(story)

    return buf.getvalue()


st.download_button(
    "⬇ Download Retinal Assessment Report",
    data=build_pdf(),
    file_name="RetinaAI_Clinical_Report.pdf",
    mime="application/pdf",
    type="primary"
)