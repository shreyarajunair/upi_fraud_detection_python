"""
UPI Fraud Detection — Streamlit Dashboard
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL  = "http://127.0.0.1:5000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
    .fraud-card  { background:#FF4B4B22; border:1px solid #FF4B4B;
                   border-radius:10px; padding:12px; margin:8px 0; }
    .safe-card   { background:#00CC9622; border:1px solid #00CC96;
                   border-radius:10px; padding:12px; margin:8px 0; }
    .block-label { color:#FF4B4B; font-weight:bold; font-size:16px; }
    .allow-label { color:#00CC96; font-weight:bold; font-size:16px; }
    .model-badge { background:#1E3A5F; color:#7EC8E3; padding:4px 10px;
                   border-radius:20px; font-size:13px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── API helpers ────────────────────────────────────────────
def check_api():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except:
        return False

def get_model_info():
    try:
        return requests.get(f"{API_URL}/model-info", timeout=3).json()
    except:
        return {}

def predict_transaction(data):
    try:
        return requests.post(f"{API_URL}/predict", json=data, timeout=10).json()
    except Exception as e:
        return {"error": str(e)}

def get_fraud_log():
    try:
        return requests.get(f"{API_URL}/fraud-log", timeout=5).json()
    except:
        return {"cases": [], "total": 0}

def get_fraud_stats():
    try:
        return requests.get(f"{API_URL}/fraud-stats", timeout=5).json()
    except:
        return {}

def get_history():
    try:
        return requests.get(f"{API_URL}/history", timeout=5).json()
    except:
        return []

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size:52px;text-align:center'>₹</h1>",
                unsafe_allow_html=True)
    st.title("UPI Fraud Detection")
    st.caption("Powered by Deep Learning")
    st.divider()

    api_ok = check_api()
    if api_ok:
        st.success("API: Online ✅")
        info = get_model_info()
        if info:
            st.markdown(
                f"<span class='model-badge'>🤖 {info.get('best_model','—')}</span>",
                unsafe_allow_html=True
            )
            st.caption(
                f"AUC: {info.get('auc','—')} | "
                f"Recall: {info.get('recall','—')} | "
                f"Threshold: {info.get('threshold','—')}"
            )
    else:
        st.error("API: Offline ❌")
        st.code("cd api\npython app.py")
        st.stop()

    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Model Comparison",
        "🔍 Check Transaction",
        "💳 UPI Simulator",
        "📋 Fraud Log",
        "📈 Analytics"
    ], label_visibility="hidden")

    st.divider()
    st.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh"):
        st.rerun()

# ══════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛡️ UPI Fraud Detection System")
    st.caption("Real-time fraud monitoring powered by deep learning")
    st.divider()

    stats   = get_fraud_stats()
    history = get_history()
    info    = get_model_info()

    if info:
        st.info(
            f"**Active Model:** {info.get('best_model','—')}  |  "
            f"ROC-AUC: **{info.get('auc','—')}**  |  "
            f"Recall: **{info.get('recall','—')}**  |  "
            f"Threshold: **{info.get('threshold','—')}**"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Fraud Cases (DB)", stats.get('total_fraud_cases', 0))
    with col2:
        st.metric("Avg Fraud Amount", f"₹{stats.get('avg_fraud_amount', 0):,.0f}")
    with col3:
        st.metric("Avg Fraud Probability",
                  f"{stats.get('avg_fraud_probability', 0)*100:.1f}%")
    with col4:
        total_h = len(history)
        fraud_h = sum(1 for t in history if t.get('result', {}).get('is_fraud'))
        rate    = fraud_h / total_h * 100 if total_h else 0
        st.metric("Session Fraud Rate", f"{rate:.1f}%",
                  delta=f"{total_h} transactions")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud by Risk Level")
        by_risk = stats.get('by_risk_level', {})
        if by_risk:
            fig = px.pie(
                names=list(by_risk.keys()),
                values=list(by_risk.values()),
                color=list(by_risk.keys()),
                color_discrete_map={
                    'HIGH':'#FF4B4B','MEDIUM':'#FFA500',
                    'LOW':'#FFD700','SAFE':'#00CC96'
                },
                hole=0.4
            )
            fig.update_layout(height=280, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud cases yet.")

    with col2:
        st.subheader("Fraud by Transaction Type")
        by_type = stats.get('by_transaction_type', {})
        if by_type:
            fig = px.bar(
                x=list(by_type.keys()),
                y=list(by_type.values()),
                color=list(by_type.values()),
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=280, showlegend=False,
                              margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud cases yet.")

    st.subheader("Recent Transactions (This Session)")
    if history:
        for t in reversed(history[-6:]):
            result   = t.get('result', {})
            is_fraud = result.get('is_fraud', False)
            card_cls = "fraud-card" if is_fraud else "safe-card"
            lbl_cls  = "block-label" if is_fraud else "allow-label"
            st.markdown(f"""
            <div class="{card_cls}">
                <span class="{lbl_cls}">{result.get('action','—')}</span> &nbsp;|&nbsp;
                ₹{t.get('amount', 0):,} &nbsp;|&nbsp;
                {t.get('sender_bank','—')} &nbsp;|&nbsp;
                Prob: <b>{result.get('fraud_probability',0)*100:.1f}%</b> &nbsp;|&nbsp;
                Risk: <b>{result.get('risk_level','—')}</b> &nbsp;|&nbsp;
                Model: <b>{result.get('model_used','—')}</b>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No transactions this session yet.")

# ══════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison — All 6 Algorithms")
    st.caption("Results from 05_deep_learning_comparison.ipynb")
    st.divider()

    # Hardcoded results from your actual notebook output
    comparison_data = {
        'Model':       ['Sequential LSTM-CNN', 'XGBoost', 'Vanilla LSTM-CNN',
                        'SVM', 'K-Means', 'Random Forest'],
        'ROC-AUC':     [0.9999, 0.9998, 0.9998, 0.9998, 0.9995, 0.9996],
        'Precision':   [0.3200, 0.4545, 0.3684, 0.1944, 0.4000, 0.2222],
        'Recall':      [1.0000, 0.6250, 0.8750, 0.8750, 0.7500, 0.5000],
        'F1-Score':    [0.4848, 0.5263, 0.5185, 0.3182, 0.5217, 0.3077],
        'Fraud Caught':  [8, 5, 7, 7, 6, 4],
        'Total Fraud':   [8, 8, 8, 8, 8, 8],
        'False Alarms':  [17, 6, 12, 29, 9, 14],
    }
    df_cmp = pd.DataFrame(comparison_data)
    df_cmp['Catch Rate'] = (
        df_cmp['Fraud Caught'] / df_cmp['Total Fraud'] * 100
    ).round(1).astype(str) + '%'

    # Winner banner
    st.success(
        "🏆 **Winner: Sequential LSTM-CNN with Attention** — "
        "ROC-AUC: 0.9999 | Recall: 100% | Fraud caught: 8/8"
    )
    st.markdown("""
    > **Why Sequential LSTM-CNN wins:** The bidirectional LSTM captures patterns
    > in both directions across features, while the attention mechanism learns
    > to focus specifically on the most suspicious feature combinations
    > (amount × time × network × device). This gives it perfect recall —
    > catching every fraud case — while maintaining a reasonable false alarm rate.
    """)

    st.divider()

    # Colour coded table
    def colour_recall(val):
        if val >= 0.8:  return 'background-color: #00CC9644'
        if val >= 0.6:  return 'background-color: #FFD70044'
        return 'background-color: #FF4B4B44'

    styled = df_cmp.style.map(colour_recall, subset=['Recall'])
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # Interactive bar charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC-AUC Score")
        fig = px.bar(
            df_cmp, x='ROC-AUC', y='Model', orientation='h',
            color='ROC-AUC', color_continuous_scale='Blues',
            range_x=[0.998, 1.0]
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recall (Fraud Detection Rate)")
        fig = px.bar(
            df_cmp, x='Recall', y='Model', orientation='h',
            color='Recall', color_continuous_scale='Reds',
            range_x=[0, 1.1]
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud Cases Caught (out of 8)")
        fig = px.bar(
            df_cmp, x='Fraud Caught', y='Model', orientation='h',
            color='Fraud Caught', color_continuous_scale='Greens',
            range_x=[0, 9]
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("False Alarms")
        fig = px.bar(
            df_cmp, x='False Alarms', y='Model', orientation='h',
            color='False Alarms', color_continuous_scale='Oranges'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.divider()
    st.subheader("Performance Radar")
    fig = go.Figure()
    colors_list = ['#E63946','#F4A261','#2A9D8F','#457B9D','#9B2226','#6A0572']
    metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
    for i, row in df_cmp.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        cats = metrics + [metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself',
            name=row['Model'],
            line_color=colors_list[i % len(colors_list)],
            opacity=0.7
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Why less business rules
    st.divider()
    st.subheader("How This Reduces Dependence on Business Rules")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Old system (Random Forest + heavy rules):**
        - ML model: 0.49 ROC-AUC (random guessing)
        - Business rules: doing 90% of the work
        - 34/96 fraud caught (35% recall)
        - Rules needed to override ML completely
        """)
    with col2:
        st.markdown("""
        **New system (Sequential LSTM-CNN + minimal rules):**
        - ML model: 0.9999 ROC-AUC (near perfect)
        - Business rules: only 2 extreme edge-case nudges
        - 8/8 fraud caught (100% recall)
        - Rules add ≤0.12 nudge at most
        """)

    # Check if comparison images exist from notebook
    roc_img = os.path.join(BASE_DIR, '..', 'data', 'roc_comparison.png')
    cmp_img = os.path.join(BASE_DIR, '..', 'data', 'model_comparison.png')

    if os.path.exists(roc_img) or os.path.exists(cmp_img):
        st.divider()
        st.subheader("Charts from Notebook")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curves — All 6 Models",
                     use_container_width=True)
        if os.path.exists(cmp_img):
            st.image(cmp_img, caption="Model Comparison Chart",
                     use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: Check Transaction
# ══════════════════════════════════════════════════════════
elif page == "🔍 Check Transaction":
    st.title("🔍 Check a Transaction")
    st.caption("Enter transaction details to get a real-time fraud prediction")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        amount       = st.number_input("Amount (INR) ₹", 10, 200000, 5000)
        txn_type     = st.selectbox("Transaction Type",
                                    ["P2P","P2M","Bill Payment","Recharge"])
        merch_cat    = st.selectbox("Merchant Category", [
            "Grocery","Electronics","Food & Dining","Travel","Healthcare",
            "Education","Entertainment","Utilities","Shopping","Insurance",
            "Investment","Fuel","Real Estate","Government","Other"
        ])
        sender_bank  = st.selectbox("Sender Bank", [
            "HDFC","SBI","ICICI","Axis","Yes Bank","Kotak","IndusInd",
            "Punjab National Bank","Bank of Baroda","Canara Bank",
            "Union Bank","IDFC First","Federal Bank","South Indian Bank",
            "Bandhan Bank","RBL Bank","Indian Bank","UCO Bank",
            "Bank of India","Central Bank of India"
        ])
        receiver_bank = st.selectbox("Receiver Bank", [
            "HDFC","SBI","ICICI","Axis","Yes Bank","Kotak","IndusInd",
            "Punjab National Bank","Bank of Baroda","Canara Bank",
            "Union Bank","IDFC First","Federal Bank","South Indian Bank",
            "Bandhan Bank","RBL Bank","Indian Bank","UCO Bank",
            "Bank of India","Central Bank of India"
        ])
        sender_state = st.selectbox("Sender State", [
            "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
            "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka",
            "Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya",
            "Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim",
            "Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand",
            "West Bengal","Delhi","Jammu & Kashmir","Ladakh","Puducherry",
            "Chandigarh","Dadra & Nagar Haveli","Lakshadweep",
            "Andaman & Nicobar Islands"
        ])

    with col2:
        hour_of_day      = st.slider("Hour of Day", 0, 23, 14)
        device_type      = st.selectbox("Device Type", ["Android","iOS","Web"])
        network_type     = st.selectbox("Network Type", ["4G","5G","WiFi","3G"])
        sender_age_grp   = st.selectbox("Sender Age Group",
                                        ["18-25","26-35","36-45","46-55","56+"])
        receiver_age_grp = st.selectbox("Receiver Age Group",
                                        ["18-25","26-35","36-45","46-55","56+"])
        day_of_week      = st.selectbox("Day of Week", [
            "Monday","Tuesday","Wednesday","Thursday","Friday",
            "Saturday","Sunday"
        ])
        is_weekend = st.checkbox("Is Weekend?")
        st.caption("ℹ️ High risk: night hours (0–4), large amounts, WiFi, Web device")

    st.divider()
    if st.button("🔍 Check for Fraud", type="primary", use_container_width=True):
        payload = {
            "transaction type":   txn_type,
            "merchant_category":  merch_cat,
            "amount (INR)":       amount,
            "sender_age_group":   sender_age_grp,
            "receiver_age_group": receiver_age_grp,
            "sender_state":       sender_state,
            "sender_bank":        sender_bank,
            "receiver_bank":      receiver_bank,
            "device_type":        device_type,
            "network_type":       network_type,
            "hour_of_day":        hour_of_day,
            "day_of_week":        day_of_week,
            "is_weekend":         is_weekend
        }
        with st.spinner("Analysing transaction..."):
            result = predict_transaction(payload)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                if result['is_fraud']:
                    st.error(f"🚨 {result['action']}")
                else:
                    st.success(f"✅ {result['action']}")
            with c2:
                st.metric("Fraud Probability",
                          f"{result['fraud_probability']*100:.1f}%")
            with c3:
                icons = {'HIGH':'🔴','MEDIUM':'🟠','LOW':'🟡','SAFE':'🟢'}
                risk  = result['risk_level']
                st.metric("Risk Level", f"{icons.get(risk,'')} {risk}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['fraud_probability'] * 100,
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis':  {'range': [0, 100]},
                    'bar':   {'color': 'darkred'},
                    'steps': [
                        {'range':[0,10],   'color':'#00CC96'},
                        {'range':[10,30],  'color':'#FFD700'},
                        {'range':[30,60],  'color':'#FFA500'},
                        {'range':[60,100], 'color':'#FF4B4B'}
                    ]
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Message:** {result['message']}")
            st.markdown(f"**Model used:** `{result.get('model_used','—')}`")
            st.markdown(
                f"**Rules triggered:** "
                f"{', '.join(result.get('triggered_rules', []))}"
            )

# ══════════════════════════════════════════════════════════
# PAGE: UPI Simulator
# ══════════════════════════════════════════════════════════
elif page == "💳 UPI Simulator":
    sim_path = os.path.join(BASE_DIR, 'upi_simulator.py')
    if os.path.exists(sim_path):
        exec(open(sim_path, encoding='utf-8').read())
    else:
        st.warning("upi_simulator.py not found in dashboard/ folder.")

# ══════════════════════════════════════════════════════════
# PAGE: Fraud Log
# ══════════════════════════════════════════════════════════
elif page == "📋 Fraud Log":
    st.title("📋 Fraud Case Log")
    st.caption("All flagged transactions stored in the database")
    st.divider()

    log   = get_fraud_log()
    cases = log.get('cases', [])
    st.metric("Total Fraud Cases in Database", log.get('total', 0))

    if cases:
        df = pd.DataFrame(cases)
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            ['HIGH','MEDIUM','LOW'],
            default=['HIGH','MEDIUM','LOW']
        )
        if risk_filter:
            df = df[df['risk_level'].isin(risk_filter)]

        def colour_risk(val):
            c = {'HIGH':'background-color: #FF4B4B44','MEDIUM':'background-color: #FFA50044','LOW':'background-color: #FFD70044'}
            return c.get(val, '')

        show_cols = [c for c in [
            'timestamp','amount','transaction_type','sender_bank',
            'sender_state','device_type','network_type',
            'fraud_probability','risk_level','action'
        ] if c in df.columns]

        st.dataframe(
            df[show_cols].style.map(colour_risk, subset=['risk_level']),
            use_container_width=True, height=420
        )
        st.download_button(
            "⬇️ Download CSV", df.to_csv(index=False),
            f"fraud_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    else:
        st.info("No fraud cases yet. Test a transaction in 'Check Transaction'.")

# ══════════════════════════════════════════════════════════
# PAGE: Analytics
# ══════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.title("📈 Fraud Analytics")
    st.caption("Visual analysis of detected fraud patterns")
    st.divider()

    log   = get_fraud_log()
    cases = log.get('cases', [])
    if not cases:
        st.info("No fraud cases yet. Test some transactions first.")
        st.stop()

    df = pd.DataFrame(cases)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fraud by Hour of Day")
        hc = df['hour_of_day'].value_counts().sort_index()
        fig = px.bar(x=hc.index, y=hc.values,
                     labels={'x':'Hour','y':'Cases'},
                     color=hc.values, color_continuous_scale='Reds')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Fraud by Device Type")
        dc = df['device_type'].value_counts()
        fig = px.pie(names=dc.index, values=dc.values, hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Amount Distribution")
        fig = px.histogram(df, x='amount', nbins=25,
                           color_discrete_sequence=['#FF4B4B'],
                           labels={'amount':'Amount (INR)'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Fraud Probability Distribution")
        fig = px.histogram(df, x='fraud_probability', nbins=25,
                           color_discrete_sequence=['#FFA500'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud by Bank")
    bc = df['sender_bank'].value_counts()
    fig = px.bar(x=bc.index, y=bc.values,
                 labels={'x':'Bank','y':'Cases'},
                 color=bc.values, color_continuous_scale='Oranges')
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)