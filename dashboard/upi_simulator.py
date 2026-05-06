import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="UPI Simulator", page_icon="₹", layout="centered")

API_URL = "http://127.0.0.1:5000"

st.markdown("""
<style>
.upi-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 20px;
    padding: 30px;
    color: white;
    text-align: center;
}
.pay-button {
    background: #00C853;
    color: white;
    border: none;
    border-radius: 25px;
    padding: 15px 40px;
    font-size: 18px;
    cursor: pointer;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Simulated UPI Interface ────────────────────────────────
st.markdown("<div class='upi-card'>", unsafe_allow_html=True)
st.markdown("## ₹ UPI Payment")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    sender_upi  = st.text_input("Your UPI ID", value="shreya@hdfcbank")
    sender_bank = st.selectbox("Your Bank", [
        "HDFC", "SBI", "ICICI", "Axis", "Kotak"
    ])
    sender_age  = st.selectbox("Age Group", [
        "18-25", "26-35", "36-45", "46-55", "56+"
    ])

with col2:
    receiver_upi  = st.text_input("Pay To (UPI ID)", value="merchant@sbi")
    receiver_bank = st.selectbox("Receiver Bank", [
        "SBI", "HDFC", "ICICI", "Axis", "Kotak"
    ])
    receiver_age  = st.selectbox("Receiver Age Group", [
        "18-25", "26-35", "36-45", "46-55", "56+"
    ])

st.divider()

amount   = st.number_input("Amount ₹", min_value=1, max_value=200000, value=500)
note     = st.text_input("Payment Note (optional)", value="Grocery payment")
txn_type = st.selectbox("Payment Type", ["P2P", "P2M", "Bill Payment", "Recharge"])

# Auto-detect time
now        = datetime.now()
hour       = now.hour
is_weekend = 1 if now.weekday() >= 5 else 0
day        = now.strftime("%A")

st.caption(f"🕐 Current time: {now.strftime('%I:%M %p')} | "
           f"{'Weekend' if is_weekend else 'Weekday'}")

st.divider()

if st.button("₹ PAY NOW", use_container_width=True, type="primary"):

    # Show processing spinner
    with st.spinner("🔍 Checking transaction security..."):

        payload = {
            "transaction type":   txn_type,
            "merchant_category":  "Grocery",
            "amount (INR)":       amount,
            "sender_age_group":   sender_age,
            "receiver_age_group": receiver_age,
            "sender_state":       "Delhi",
            "sender_bank":        sender_bank,
            "receiver_bank":      receiver_bank,
            "device_type":        "Android",
            "network_type":       "4G",
            "hour_of_day":        hour,
            "day_of_week":        day,
            "is_weekend":         is_weekend
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Unable to connect to fraud detection service: {str(e)}")
            st.info("Please make sure the API server is running on http://127.0.0.1:5000")
            st.stop()
        except ValueError as e:
            st.error(f"❌ Invalid response from fraud detection service: {str(e)}")
            st.stop()

    # ── Show result ────────────────────────────────────────
    st.divider()

    # Check if we have a valid fraud detection result
    if 'is_fraud' not in result:
        st.error("❌ Invalid response from fraud detection service")
        st.write("Expected fraud detection result, but got:", result)
        st.stop()

    if result['is_fraud']:
        st.error("🚨 Transaction Blocked by Fraud Detection System")
        st.markdown(f"""
        **Reason:** Suspicious transaction pattern detected

        **Fraud Probability:** {result['fraud_probability']*100:.1f}%

        **Risk Level:** {result['risk_level']}

        **What to do:** If this was a genuine payment, please:
        - Visit your bank branch
        - Call customer care: 1800-XXX-XXXX
        - Try again during business hours
        """)

        # Show which rules triggered
        rules = result.get('triggered_rules', [])
        if rules and rules != ['ML model only']:
            st.warning("**Fraud signals detected:**")
            for r in rules:
                st.markdown(f"- ⚠️ {r}")

    else:
        st.success("✅ Payment Successful!")
        st.markdown(f"""
        **₹{amount:,}** sent to **{receiver_upi}**

        **Transaction ID:** UPI{now.strftime('%Y%m%d%H%M%S')}

        **Time:** {now.strftime('%d %b %Y, %I:%M %p')}

        **Security Check:** Passed ✓ ({result['fraud_probability']*100:.1f}% risk)
        """)
        st.balloons()