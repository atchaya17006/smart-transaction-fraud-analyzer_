import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

st.set_page_config(page_title="Sentinel | Transaction Risk Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Professional blue theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#071426;--panel:#10233d;--line:rgba(147,197,253,.20);--text:#f0f7ff;--muted:#a7bdd8;--blue:#60a5fa;--cyan:#67e8f9}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(37,99,235,.22),transparent 36%),radial-gradient(ellipse at 95% 5%,rgba(34,211,238,.11),transparent 30%),var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:rgba(7,20,38,.82)}
.block-container{padding-top:1.6rem;padding-bottom:2.4rem;max-width:1500px}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important}
p,label,.stMarkdown,[data-testid="stCaptionContainer"]{color:var(--muted)}
.hero{padding:1.55rem 1.7rem;border:1px solid rgba(96,165,250,.34);border-radius:22px;background:linear-gradient(115deg,rgba(18,48,86,.98),rgba(16,35,61,.96) 60%,rgba(13,66,86,.86));box-shadow:0 18px 55px rgba(0,0,0,.22);margin-bottom:1.2rem}
.eyebrow{color:#93c5fd;font-size:.76rem;letter-spacing:.16em;font-weight:700;text-transform:uppercase}
.hero-title{color:#f4f9ff;font-family:'Space Grotesk',sans-serif;font-size:clamp(1.65rem,3vw,2.45rem);font-weight:700;line-height:1.15;margin:.45rem 0 .55rem}
.hero-sub{color:#c4d8ee;font-size:1rem;max-width:820px}
.kpi{background:linear-gradient(145deg,rgba(19,43,73,.99),rgba(12,29,51,.99));border:1px solid var(--line);border-radius:17px;padding:1.05rem 1.1rem;min-height:118px;box-shadow:0 10px 28px rgba(0,0,0,.14)}
.kpi-label{color:#b7cbe3;font-size:.82rem;font-weight:600;margin-bottom:.45rem}.kpi-value{color:#f4f9ff;font-family:'Space Grotesk',sans-serif;font-size:2rem;line-height:1.1;font-weight:700}.kpi-note{color:#91accb;font-size:.75rem;margin-top:.45rem}
.pill{display:inline-block;border-radius:999px;padding:.28rem .65rem;font-size:.75rem;font-weight:700;border:1px solid var(--line);color:#bdeafe;background:rgba(37,99,235,.12)}
div[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(19,43,73,.99),rgba(12,29,51,.99));border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem}
div[data-testid="stMetricLabel"] p{color:#b7cbe3!important;font-weight:600}div[data-testid="stMetricValue"]{color:#f4f9ff!important;font-family:'Space Grotesk',sans-serif}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line)!important}
.stButton>button{border:1px solid rgba(125,211,252,.45);border-radius:11px;background:linear-gradient(100deg,#2563eb,#0891b2);color:white;font-weight:700;min-height:2.7rem;transition:all .18s ease}
.stButton>button:hover{border-color:#bae6fd;color:white;transform:translateY(-1px);box-shadow:0 8px 22px rgba(37,99,235,.25)}
.stDownloadButton>button{border:1px solid rgba(125,211,252,.38);border-radius:10px;background:rgba(37,99,235,.14);color:#eaf6ff;font-weight:600}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"],.stMultiSelect div[data-baseweb="select"]{border-radius:10px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}hr{border-color:var(--line)}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#10233d,#081629);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#f4f9ff!important}
.sidebar-brand{padding:.4rem 0 1rem}.sidebar-logo{font-family:'Space Grotesk',sans-serif;color:#f4f9ff;font-weight:700;font-size:1.35rem}.sidebar-caption{color:#a7bdd8;font-size:.78rem}.small-muted{color:#9bb4d1;font-size:.82rem}div[data-testid="stAlert"]{border-radius:12px}
@media(max-width:768px){.block-container{padding-left:1rem;padding-right:1rem;padding-top:1rem}.hero{padding:1.1rem;border-radius:16px}.kpi{min-height:105px;padding:.85rem}.kpi-value{font-size:1.55rem}}
</style>
""", unsafe_allow_html=True)

# Synthetic training data for educational demonstration only
random.seed(42)
training_data=[]
for _ in range(1000):
    amount=random.randint(100,100000); frequency=random.randint(1,10); new_device=random.randint(0,1); unusual_location=random.randint(0,1)
    demo_score=(amount/100000)*35+(frequency/10)*20+new_device*25+unusual_location*20
    training_data.append({"Amount":amount,"Recent Frequency":frequency,"New Device":new_device,"Unusual Location":unusual_location,"Demo Label":1 if demo_score>=65 else 0})
training_df=pd.DataFrame(training_data)
features=["Amount","Recent Frequency","New Device","Unusual Location"]
X=training_df[features]; y=training_df["Demo Label"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")
model.fit(X_train,y_train)
test_probabilities=model.predict_proba(X_test)[:,1]; test_predictions=model.predict(X_test)
precision=precision_score(y_test,test_predictions,zero_division=0); recall=recall_score(y_test,test_predictions,zero_division=0); f1=f1_score(y_test,test_predictions,zero_division=0); pr_auc=average_precision_score(y_test,test_probabilities)

def predict_risk(amount,frequency,new_device,unusual_location):
    row=pd.DataFrame([{"Amount":amount,"Recent Frequency":frequency,"New Device":new_device,"Unusual Location":unusual_location}])
    score=int(round(model.predict_proba(row)[0][1]*100))
    level="High" if score>=65 else "Medium" if score>=35 else "Low"
    return score,level

if "transactions" not in st.session_state:
    rows=[]
    for i in range(1,31):
        amount=random.randint(100,100000); frequency=random.randint(1,10); new_device=random.randint(0,1); unusual_location=random.randint(0,1)
        risk,level=predict_risk(amount,frequency,new_device,unusual_location)
        rows.append({"Transaction ID":f"TXN{i:03}","Amount":amount,"Recent Frequency":frequency,"New Device":new_device,"Unusual Location":unusual_location,"Risk Score":risk,"Risk Level":level,"Review Status":"Pending"})
    st.session_state.transactions=pd.DataFrame(rows)
df=st.session_state.transactions.copy()

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="sidebar-logo">◈ SENTINEL</div><div class="sidebar-caption">Transaction Risk Intelligence</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    page=st.radio("WORKSPACE",["Overview","Risk Simulator","Transaction Monitor","Model Insights","Review Queue"])
    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    st.markdown('<span class="pill">● Demo environment active</span>',unsafe_allow_html=True)
    st.caption("Prototype build 1.0")
    st.caption("Synthetic demonstration data")

st.markdown('''<div class="hero"><div class="eyebrow">CYBER RISK • TRANSACTION INTELLIGENCE</div><div class="hero-title">Smart Transaction<br>Fraud Risk Analyzer</div><div class="hero-sub">A machine-learning demonstration dashboard to assess transaction risk, surface review indicators, and prioritize investigation.</div></div>''',unsafe_allow_html=True)
st.info("Demo notice: this prototype uses synthetic training labels and sample transactions. A risk score is not proof of fraud and must not be treated as a real financial decision.")

total=len(df); high=int((df["Risk Level"]=="High").sum()); medium=int((df["Risk Level"]=="Medium").sum()); low=int((df["Risk Level"]=="Low").sum()); review_pending=int((df["Review Status"]=="Pending").sum())

if page=="Overview":
    st.markdown("### Overview"); st.caption("Live snapshot of the current demonstration transaction set.")
    c1,c2,c3,c4=st.columns(4)
    cards=[("TOTAL TRANSACTIONS",total,"Sample records in this session","#f4f9ff"),("HIGH RISK",high,"Prioritize for review","#fb7185"),("MEDIUM RISK",medium,"Needs closer inspection","#fbbf24"),("PENDING REVIEWS",review_pending,"Awaiting analyst review","#67e8f9")]
    for col,(label,value,note,color) in zip([c1,c2,c3,c4],cards):
        with col: st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
    left,right=st.columns([1.15,1])
    with left:
        st.markdown("### Risk distribution")
        dist=df["Risk Level"].value_counts().reindex(["High","Medium","Low"],fill_value=0)
        st.bar_chart(pd.DataFrame({"Transactions":dist.values},index=dist.index),use_container_width=True)
    with right:
        st.markdown("### Risk snapshot")
        summary=pd.DataFrame({"Risk level":["High","Medium","Low"],"Count":[high,medium,low],"Share (%)":[round(high/total*100,1) if total else 0,round(medium/total*100,1) if total else 0,round(low/total*100,1) if total else 0]})
        st.dataframe(summary,use_container_width=True,hide_index=True)
        st.markdown("### Highest-priority records")
        st.dataframe(df.sort_values("Risk Score",ascending=False)[["Transaction ID","Risk Score","Risk Level","Review Status"]].head(5),use_container_width=True,hide_index=True)
    st.markdown("### How to use this workspace")
    a,b,c=st.columns(3)
    with a: st.markdown("**01 · Simulate**"); st.caption("Change transaction attributes and inspect the model's estimated risk.")
    with b: st.markdown("**02 · Monitor**"); st.caption("Filter and sort sample transactions by risk and review status.")
    with c: st.markdown("**03 · Review**"); st.caption("Update the review queue to demonstrate an analyst workflow.")

elif page=="Risk Simulator":
    st.markdown("### 🧪 What-if risk simulator"); st.caption("Adjust transaction characteristics to see how the demonstration model responds.")
    left,right=st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### Transaction details")
            amount=st.number_input("Transaction amount (₹)",min_value=1,max_value=1000000,value=5000,step=500)
            frequency=st.slider("Recent transaction frequency",1,10,3)
            new_device=st.selectbox("New device detected?",["No","Yes"])
            unusual_location=st.selectbox("Unusual location detected?",["No","Yes"])
            analyze=st.button("Analyze transaction",use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("#### Risk assessment")
            if analyze:
                risk,level=predict_risk(amount,frequency,int(new_device=="Yes"),int(unusual_location=="Yes"))
                color="#fb7185" if level=="High" else "#fbbf24" if level=="Medium" else "#34d399"
                st.markdown(f'<div class="kpi"><div class="kpi-label">MODEL RISK SCORE</div><div class="kpi-value" style="color:{color}">{risk}<span style="font-size:1rem;color:#a7bdd8"> / 100</span></div><div class="kpi-note">Estimated category: {level.upper()}</div></div>',unsafe_allow_html=True)
                st.progress(risk/100)
                reasons=[]
                if amount>20000: reasons.append("Amount exceeds the demo review threshold.")
                if frequency>=7: reasons.append("Recent transaction frequency is high.")
                if new_device=="Yes": reasons.append("A new device is selected.")
                if unusual_location=="Yes": reasons.append("An unusual location is selected.")
                st.markdown("**Indicators to review**")
                if reasons:
                    for reason in reasons: st.markdown(f"- {reason}")
                else: st.caption("No listed demo indicators were triggered.")
                st.warning("This is a synthetic-data model estimate, not confirmation of fraud.")
            else:
                st.markdown("Choose the transaction details and select **Analyze transaction** to view a risk estimate.")
                st.caption("Tip for the demo: change one input at a time and compare the results.")

elif page=="Transaction Monitor":
    st.markdown("### 🚨 Transaction monitor"); st.caption("Explore the sample transactions and prioritize records for review.")
    f1,f2,f3=st.columns(3)
    with f1: risk_filter=st.multiselect("Risk level",["High","Medium","Low"],default=["High","Medium","Low"])
    with f2: status_filter=st.multiselect("Review status",["Pending","Under Review","Reviewed"],default=["Pending","Under Review","Reviewed"])
    with f3: search_id=st.text_input("Search transaction ID",placeholder="e.g. TXN005")
    filtered=df[df["Risk Level"].isin(risk_filter)&df["Review Status"].isin(status_filter)]
    if search_id.strip(): filtered=filtered[filtered["Transaction ID"].str.contains(search_id.strip(),case=False,na=False)]
    st.markdown(f"**{len(filtered)} transaction(s) match your filters**")
    st.dataframe(filtered.sort_values("Risk Score",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("Download filtered transactions (CSV)",data=filtered.to_csv(index=False).encode("utf-8"),file_name="transaction_risk_demo.csv",mime="text/csv")

elif page=="Model Insights":
    st.markdown("### 🧠 Model insights"); st.caption("Evaluation results show agreement with generated demonstration labels only.")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Precision",f"{precision:.2f}"); m2.metric("Recall",f"{recall:.2f}"); m3.metric("F1-score",f"{f1:.2f}"); m4.metric("PR-AUC",f"{pr_auc:.2f}")
    st.warning("These metrics do not establish real-world fraud-detection performance because the model is trained and tested on synthetic labels.")
    left,right=st.columns(2)
    with left:
        st.markdown("#### Feature importance")
        importance=pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
        st.bar_chart(importance.set_index("Feature"),use_container_width=True)
    with right:
        st.markdown("#### Model configuration")
        st.dataframe(pd.DataFrame({"Setting":["Algorithm","Number of trees","Training samples","Test samples","Label source"],"Value":["Random Forest Classifier","100",str(len(X_train)),str(len(X_test)),"Synthetic demonstration rules"]}),use_container_width=True,hide_index=True)
        st.caption("Feature importance describes the model's use of inputs; it does not prove causation.")

elif page=="Review Queue":
    st.markdown("### 📝 Investigation review queue"); st.caption("Demo workflow for recording review progress. This is not a live banking system.")
    ids=df.sort_values("Risk Score",ascending=False)["Transaction ID"].tolist()
    selected_id=st.selectbox("Select transaction",ids)
    current_status=st.session_state.transactions.loc[st.session_state.transactions["Transaction ID"]==selected_id,"Review Status"].iloc[0]
    new_status=st.selectbox("Set review status",["Pending","Under Review","Reviewed"],index=["Pending","Under Review","Reviewed"].index(current_status))
    if st.button("Save review status"):
        st.session_state.transactions.loc[st.session_state.transactions["Transaction ID"]==selected_id,"Review Status"]=new_status
        st.success(f"{selected_id} status updated to {new_status}.")
        st.rerun()
    st.markdown("#### Current queue")
    queue=st.session_state.transactions.sort_values("Risk Score",ascending=False)
    st.dataframe(queue[["Transaction ID","Amount","Risk Score","Risk Level","Review Status"]],use_container_width=True,hide_index=True)
    st.download_button("Download review queue (CSV)",data=queue.to_csv(index=False).encode("utf-8"),file_name="investigation_review_queue.csv",mime="text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">SENTINEL · Smart Transaction Fraud Risk Analyzer &nbsp;|&nbsp; Educational prototype · Synthetic demonstration data only</div>',unsafe_allow_html=True)
