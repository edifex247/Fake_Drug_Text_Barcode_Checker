import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from pyzbar.pyzbar import decode


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Fake Drug Checker",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# CUSTOM STYLING
# ==========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

.hero {
    padding: 1.8rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #eef7ff 0%,
        #f8fbff 100%
    );
    border: 1px solid #d9eaf7;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 750;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #52606d;
}

.section-card {
    padding: 1.2rem;
    border-radius: 14px;
    border: 1px solid #e1e7ec;
    background: white;
    margin-bottom: 1rem;
}

.small-note {
    color: #667085;
    font-size: 0.9rem;
}

.footer {
    text-align: center;
    color: #667085;
    font-size: 0.85rem;
    padding: 2rem 0 1rem 0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# FILE PATHS
# ==========================================================

MODEL_PATH = "model/drug_checker_model.pkl"
REFERENCE_PATH = "data/reference_products.csv"
NAFDAC_PATH = "data/nafdac_data.csv"
NAFDAC_ALERTS_PATH = "data/nafdac_alerts.csv"


# ==========================================================
# LOAD MODEL
# ==========================================================

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


# ==========================================================
# LOAD REFERENCE DATABASE
# ==========================================================

try:
    reference_df = pd.read_csv(REFERENCE_PATH)
    reference_loaded = True
except Exception:
    reference_df = pd.DataFrame()
    reference_loaded = False


# ==========================================================
# LOAD NAFDAC DATABASE
# ==========================================================

try:
    nafdac_df = pd.read_csv(NAFDAC_PATH)
    nafdac_loaded = True
except Exception:
    nafdac_df = pd.DataFrame()
    nafdac_loaded = False


# ==========================================================
# LOAD NAFDAC ALERT DATABASE
# ==========================================================

try:
    nafdac_alerts_df = pd.read_csv(NAFDAC_ALERTS_PATH)
    nafdac_alerts_loaded = True
except Exception:
    nafdac_alerts_df = pd.DataFrame()
    nafdac_alerts_loaded = False


# ==========================================================
# BARCODE SCANNER
# ==========================================================

def scan_barcode(uploaded_file):

    try:

        file_bytes = uploaded_file.getvalue()

        image_array = cv2.imdecode(
            np.frombuffer(
                file_bytes,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if image_array is None:
            return None

        detected_barcodes = decode(image_array)

        if detected_barcodes:

            return detected_barcodes[0].data.decode(
                "utf-8"
            )

        return None

    except Exception:
        return None


# ==========================================================
# HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
💊 Fake Drug Text/Barcode Checker
</div>

<div class="hero-subtitle">
AI-Assisted Medicine Verification Prototype
</div>

<br>

An AI/ML prototype that combines drug information,
barcode recognition, reference data and regulatory
alert information to identify potentially suspicious products.

</div>
""", unsafe_allow_html=True)


st.warning(
    "⚠️ Prototype only. This system does not replace official "
    "NAFDAC verification, laboratory testing, or professional "
    "pharmaceutical advice."
)


# ==========================================================
# HOW IT WORKS
# ==========================================================

with st.expander("ℹ️ How this prototype works"):

    st.markdown("""
### Four verification layers

**1️⃣ NAFDAC Regulatory Alerts**

Checks whether the supplied product or NAFDAC number
appears in the loaded regulatory-alert dataset.

**2️⃣ NAFDAC Reference Database**

Checks supplied drug information against the loaded
NAFDAC reference records.

**3️⃣ Prototype Reference Database**

Checks against the demonstration product database.

**4️⃣ Machine Learning Assessment**

Uses the trained prototype ML model to assess whether
the supplied information is more consistent with the
positive or suspicious training examples.

### Result priority

🔴 Known NAFDAC Alert  
↓  
🟢 NAFDAC Reference Match  
↓  
🟢 Prototype Reference Match  
↓  
🔴 Potentially Suspicious  
↓  
🟠 Needs Further Verification
""")


# ==========================================================
# INPUT SECTION
# ==========================================================

st.markdown("## 🔎 Step 1 — Enter Drug Information")

col1, col2 = st.columns(2)

with col1:

    drug_name = st.text_input(
        "Drug Name",
        placeholder="e.g. Paracetamol"
    )

    nafdac_number = st.text_input(
        "NAFDAC Number",
        placeholder="e.g. A11-100860"
    )

    batch_number = st.text_input(
        "Batch Number",
        placeholder="Enter batch number"
    )


with col2:

    manufacturer = st.text_input(
        "Manufacturer",
        placeholder="Enter manufacturer"
    )

    barcode = st.text_input(
        "Barcode",
        placeholder="Enter barcode manually"
    )


# ==========================================================
# BARCODE UPLOAD
# ==========================================================

st.markdown("## 📦 Step 2 — Barcode Verification")

st.caption(
    "Enter a barcode manually or upload a clear image of the barcode."
)

uploaded_barcode = st.file_uploader(
    "📷 Upload barcode image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_barcode is not None:

    st.image(
        uploaded_barcode,
        caption="Uploaded barcode",
        width=450
    )

    scanned_barcode = scan_barcode(
        uploaded_barcode
    )

    if scanned_barcode:

        st.success(
            f"✅ Barcode detected: {scanned_barcode}"
        )

        barcode = scanned_barcode

    else:

        st.warning(
            "⚠️ No barcode could be detected. "
            "Try uploading a clearer image."
        )


st.divider()


# ==========================================================
# CHECK BUTTON
# ==========================================================

check_button = st.button(
    "🔍 Check Drug",
    use_container_width=True,
    type="primary"
)


if check_button:

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    if not drug_name and not nafdac_number and not barcode:

        st.warning(
            "Please enter at least a Drug Name, "
            "NAFDAC Number, or Barcode."
        )

    else:

        # ==================================================
        # NORMALIZE INPUT
        # ==================================================

        drug_input = drug_name.strip().lower()
        nafdac_input = nafdac_number.strip().lower()
        batch_input = batch_number.strip().lower()
        manufacturer_input = manufacturer.strip().lower()
        barcode_input = barcode.strip()


        # ==================================================
        # NAFDAC ALERT CHECK
        # ==================================================

        alert_match = False
        matched_alert = None

        if nafdac_alerts_loaded:

            for _, alert in nafdac_alerts_df.iterrows():

                alert_product = str(
                    alert.get("product_name", "")
                ).strip().lower()

                alert_number = str(
                    alert.get("stated_nafdac_number", "")
                ).strip().lower()

                if alert_number in ["nan", "none"]:
                    alert_number = ""

                # Exact product-name matching.
                # This prevents generic names such as
                # "Paracetamol" from matching a specific
                # product alert.

                product_match = (
                    drug_input != ""
                    and
                    drug_input == alert_product
                )

                number_match = (
                    nafdac_input != ""
                    and
                    alert_number != ""
                    and
                    nafdac_input == alert_number
                )

                if product_match or number_match:

                    alert_match = True
                    matched_alert = alert
                    break


        # ==================================================
        # NAFDAC REFERENCE CHECK
        # ==================================================

        nafdac_match = False
        nafdac_product = None

        if nafdac_loaded:

            for _, row in nafdac_df.iterrows():

                registered_drug = str(
                    row.get("drug_name", "")
                ).strip().lower()

                registered_number = str(
                    row.get("nafdac_number", "")
                ).strip().lower()

                registered_barcode = str(
                    row.get("barcode", "")
                ).strip()

                if registered_number in ["nan", "none"]:
                    registered_number = ""

                if registered_barcode.lower() in [
                    "nan",
                    "none"
                ]:
                    registered_barcode = ""

                name_match = (
                    drug_input != ""
                    and
                    drug_input == registered_drug
                )

                number_match = (
                    nafdac_input != ""
                    and
                    registered_number != ""
                    and
                    nafdac_input == registered_number
                )

                barcode_match = (
                    barcode_input != ""
                    and
                    registered_barcode != ""
                    and
                    barcode_input == registered_barcode
                )

                if (
                    (name_match and number_match)
                    or
                    barcode_match
                ):

                    nafdac_match = True
                    nafdac_product = row
                    break


        # ==================================================
        # PROTOTYPE REFERENCE CHECK
        # ==================================================

        reference_match = False
        matched_product = None

        if reference_loaded:

            for _, row in reference_df.iterrows():

                reference_drug = str(
                    row.get("drug_name", "")
                ).strip().lower()

                reference_number = str(
                    row.get("nafdac_number", "")
                ).strip().lower()

                reference_batch = str(
                    row.get("batch_number", "")
                ).strip().lower()

                reference_manufacturer = str(
                    row.get("manufacturer", "")
                ).strip().lower()

                reference_barcode = str(
                    row.get("barcode", "")
                ).strip()

                if reference_number in ["nan", "none"]:
                    reference_number = ""

                if reference_batch in ["nan", "none"]:
                    reference_batch = ""

                if reference_manufacturer in ["nan", "none"]:
                    reference_manufacturer = ""

                if reference_barcode.lower() in [
                    "nan",
                    "none"
                ]:
                    reference_barcode = ""

                name_match = (
                    drug_input != ""
                    and
                    drug_input == reference_drug
                )

                number_match = (
                    nafdac_input != ""
                    and
                    nafdac_input == reference_number
                )

                batch_match = (
                    batch_input != ""
                    and
                    batch_input == reference_batch
                )

                manufacturer_match = (
                    manufacturer_input != ""
                    and
                    manufacturer_input == reference_manufacturer
                )

                barcode_match = (
                    barcode_input != ""
                    and
                    barcode_input == reference_barcode
                )

                strong_match = (
                    barcode_match
                    or
                    (
                        name_match
                        and
                        number_match
                        and
                        batch_match
                        and
                        manufacturer_match
                    )
                )

                if strong_match:

                    reference_match = True
                    matched_product = row
                    break


        # ==================================================
        # MACHINE LEARNING ASSESSMENT
        # ==================================================

        prediction = None
        suspicious_probability = None
        genuine_probability = None

        if model_loaded:

            nafdac_valid = (
                1 if nafdac_input else 0
            )

            batch_valid = (
                1 if batch_input else 0
            )

            barcode_valid = (
                1 if barcode_input else 0
            )

            manufacturer_known = (
                1 if manufacturer_input else 0
            )

            text_complete = (
                1
                if (
                    drug_input
                    and
                    nafdac_input
                    and
                    batch_input
                    and
                    manufacturer_input
                )
                else 0
            )

            input_data = pd.DataFrame(
                [[
                    nafdac_valid,
                    batch_valid,
                    barcode_valid,
                    manufacturer_known,
                    text_complete
                ]],
                columns=[
                    "nafdac_valid",
                    "batch_valid",
                    "barcode_valid",
                    "manufacturer_known",
                    "text_complete"
                ]
            )

            try:

                prediction = model.predict(
                    input_data
                )[0]

                probabilities = model.predict_proba(
                    input_data
                )[0]

                suspicious_probability = probabilities[0]

                genuine_probability = probabilities[1]

            except Exception:

                prediction = None


        # ==================================================
        # VERIFICATION RESULT
        # ==================================================

        st.markdown("## 🔎 Verification Result")


        # ==================================================
        # NAFDAC ALERT
        # ==================================================

        if alert_match:

            st.error(
                "🔴 KNOWN NAFDAC ALERT"
            )

            st.write(
                "The supplied drug information matches "
                "an official NAFDAC regulatory alert in "
                "the loaded alert database."
            )

            if matched_alert is not None:

                st.markdown(
                    "### 🚨 NAFDAC Alert Record"
                )

                alert_product = matched_alert.get(
                    "product_name",
                    "Not stated"
                )

                alert_date = matched_alert.get(
                    "alert_date",
                    "Not stated"
                )

                classification = matched_alert.get(
                    "classification",
                    "Not stated"
                )

                alert_number = matched_alert.get(
                    "stated_nafdac_number",
                    ""
                )

                alert_batch = matched_alert.get(
                    "batch_number",
                    ""
                )

                alert_manufacturer = matched_alert.get(
                    "manufacturer",
                    ""
                )

                alert_reference = matched_alert.get(
                    "alert_reference",
                    "Not stated"
                )

                if pd.isna(alert_number) or str(
                    alert_number
                ).lower() == "nan":

                    alert_number = "Not stated"

                if pd.isna(alert_batch) or str(
                    alert_batch
                ).lower() == "nan":

                    alert_batch = "Not stated"

                if pd.isna(alert_manufacturer) or str(
                    alert_manufacturer
                ).lower() == "nan":

                    alert_manufacturer = "Not stated"

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.write(
                        f"**Product:** {alert_product}"
                    )

                    st.write(
                        f"**Alert Date:** {alert_date}"
                    )

                    st.write(
                        f"**Classification:** "
                        f"{classification}"
                    )

                    st.write(
                        f"**NAFDAC Number:** "
                        f"{alert_number}"
                    )

                with result_col2:

                    st.write(
                        f"**Batch Number:** "
                        f"{alert_batch}"
                    )

                    st.write(
                        f"**Manufacturer:** "
                        f"{alert_manufacturer}"
                    )

                    st.write(
                        f"**Alert Reference:** "
                        f"{alert_reference}"
                    )

            st.warning(
                "⚠️ This result is based on an official "
                "NAFDAC regulatory-alert record. Do not use "
                "a medicine identified in an official alert "
                "until it has been properly verified through "
                "official channels."
            )


        # ==================================================
        # NAFDAC REFERENCE
        # ==================================================

        elif nafdac_match:

            st.success(
                "🟢 NAFDAC REFERENCE MATCH"
            )

            st.write(
                "The supplied drug information matches "
                "a record in the loaded NAFDAC reference dataset."
            )

            st.info(
                "This is a reference-database match. "
                "Confirm current registration status through "
                "official NAFDAC channels."
            )

            if nafdac_product is not None:

                st.markdown(
                    "### 📋 NAFDAC Reference Record"
                )

                product_name = nafdac_product.get(
                    "drug_name",
                    "Not stated"
                )

                product_number = nafdac_product.get(
                    "nafdac_number",
                    "Not stated"
                )

                product_batch = nafdac_product.get(
                    "batch_number",
                    ""
                )

                product_manufacturer = nafdac_product.get(
                    "manufacturer",
                    ""
                )

                product_barcode = nafdac_product.get(
                    "barcode",
                    ""
                )

                product_status = nafdac_product.get(
                    "status",
                    "Listed in NAFDAC reference dataset"
                )

                if pd.isna(product_batch) or str(
                    product_batch
                ).lower() == "nan":

                    product_batch = (
                        "Not available in Greenbook data"
                    )

                if pd.isna(product_manufacturer) or str(
                    product_manufacturer
                ).lower() == "nan":

                    product_manufacturer = (
                        "Not available in Greenbook data"
                    )

                if pd.isna(product_barcode) or str(
                    product_barcode
                ).lower() == "nan":

                    product_barcode = (
                        "Not available in Greenbook data"
                    )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.write(
                        f"**Drug:** {product_name}"
                    )

                    st.write(
                        f"**NAFDAC Number:** "
                        f"{product_number}"
                    )

                    st.write(
                        f"**Batch:** {product_batch}"
                    )

                with result_col2:

                    st.write(
                        f"**Manufacturer:** "
                        f"{product_manufacturer}"
                    )

                    st.write(
                        f"**Barcode:** {product_barcode}"
                    )

                    st.write(
                        f"**Status:** {product_status}"
                    )


        # ==================================================
        # PROTOTYPE REFERENCE
        # ==================================================

        elif reference_match:

            st.success(
                "🟢 REFERENCE MATCH FOUND"
            )

            st.write(
                "The supplied information matches a product "
                "in the prototype reference database."
            )

            if genuine_probability is not None:

                st.metric(
                    "ML consistency score",
                    f"{genuine_probability * 100:.1f}%"
                )

            st.caption(
                "Prototype reference database record — "
                "not official NAFDAC verification."
            )

            if matched_product is not None:

                st.markdown(
                    "### 📋 Matched Reference Record"
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.write(
                        f"**Drug:** "
                        f"{matched_product['drug_name']}"
                    )

                    st.write(
                        f"**Barcode:** "
                        f"{matched_product['barcode']}"
                    )

                with result_col2:

                    st.write(
                        f"**Manufacturer:** "
                        f"{matched_product['manufacturer']}"
                    )


        # ==================================================
        # ML SUSPICIOUS
        # ==================================================

        elif (
            prediction is not None
            and
            prediction == 0
        ):

            st.error(
                "🔴 POTENTIALLY SUSPICIOUS"
            )

            st.write(
                "The supplied information did not match "
                "the available reference records and the "
                "ML model found it more consistent with "
                "the suspicious training examples."
            )

            st.metric(
                "Suspicion score",
                f"{suspicious_probability * 100:.1f}%"
            )


        # ==================================================
        # NEEDS VERIFICATION
        # ==================================================

        else:

            st.warning(
                "🟠 NEEDS FURTHER VERIFICATION"
            )

            st.write(
                "The product was not found in the available "
                "NAFDAC reference or alert datasets."
            )

            if genuine_probability is not None:

                st.metric(
                    "ML consistency score",
                    f"{genuine_probability * 100:.1f}%"
                )


        # ==================================================
        # INFORMATION CHECKED
        # ==================================================

        st.markdown(
            "### 📝 Information Checked"
        )

        result_table = pd.DataFrame({
            "Field": [
                "Drug Name",
                "NAFDAC Number",
                "Batch Number",
                "Manufacturer",
                "Barcode"
            ],
            "Provided": [
                drug_name
                if drug_name
                else "Not provided",

                nafdac_number
                if nafdac_number
                else "Not provided",

                batch_number
                if batch_number
                else "Not provided",

                manufacturer
                if manufacturer
                else "Not provided",

                barcode
                if barcode
                else "Not provided"
            ]
        })

        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True
        )


        # ==================================================
        # SAFETY NOTICE
        # ==================================================

        st.warning(
            "Important: This is an AI/ML prototype. It cannot "
            "confirm that a medicine is genuine or counterfeit. "
            "Products identified as suspicious or appearing in "
            "regulatory alerts should be verified through "
            "official NAFDAC channels or a qualified pharmacist."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">

<strong>Fake Drug Text/Barcode Checker</strong><br>
AI/ML Capstone Prototype • Demonstration System<br><br>
Not a substitute for official regulatory verification.

</div>
""", unsafe_allow_html=True)