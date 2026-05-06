import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Email configuration ───────────────────────────────────
# Replace with your Gmail address and App Password
SENDER_EMAIL    = "shreyarajunair@gmail.com"
SENDER_PASSWORD = "nrwzjzjexrxqxenk"   # 16-character Google App Password
ALERT_RECEIVER  = "shreyarajunair@gmail.com"  # Who receives fraud alerts


def send_fraud_alert(data, result):
    """
    Sends an email alert when a fraud transaction is detected.
    Only called when is_fraud=True.
    """
    try:
        # Build email content
        subject = f"🚨 UPI Fraud Alert — ₹{data['amount (INR)']} | {result['risk_level']} Risk"

        body = f"""
        ╔══════════════════════════════════════════╗
              UPI FRAUD DETECTION SYSTEM ALERT
        ╚══════════════════════════════════════════╝

        A suspicious UPI transaction has been detected.

        ── Transaction Details ──────────────────────
        Time:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Amount:            ₹{data['amount (INR)']}
        Transaction Type:  {data['transaction type']}
        Sender Bank:       {data['sender_bank']}
        Receiver Bank:     {data['receiver_bank']}
        Sender State:      {data['sender_state']}
        Device:            {data['device_type']}
        Network:           {data['network_type']}
        Hour of Day:       {data['hour_of_day']}:00

        ── Fraud Analysis ───────────────────────────
        Fraud Probability: {result['fraud_probability']*100:.1f}%
        Risk Level:        {result['risk_level']}
        Action Taken:      {result['action']}

        ── Recommendation ───────────────────────────
        This transaction has been automatically BLOCKED.
        Please review and contact the account holder
        if this was a legitimate transaction.

        ─────────────────────────────────────────────
        UPI Fraud Detection System
        Powered by Machine Learning
        """

        # Build the email message
        msg = MIMEMultipart()
        msg['From']    = SENDER_EMAIL
        msg['To']      = ALERT_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send via Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, ALERT_RECEIVER, msg.as_string())

        print(f"Alert email sent for ₹{data['amount (INR)']} transaction.")
        return True

    except Exception as e:
        print(f"Email alert failed: {e}")
        return False  # Don't crash the API if email fails


def should_alert(result):
    """
    Decides whether to send an email alert.
    Only alert for MEDIUM and HIGH risk — not LOW.
    Avoids alert fatigue from too many low-risk flags.
    """
    return result['risk_level'] in ['MEDIUM', 'HIGH']