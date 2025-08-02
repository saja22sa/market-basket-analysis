import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.title("تحليل قواعد الترابط من ملف Excel")

uploaded_file = st.file_uploader("ارفع ملف Excel يحتوي على بيانات التصنيفات", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # قراءة الملف
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.subheader("📊 معاينة أولية للبيانات:")
        st.dataframe(df_raw.head())

        # تنظيف البيانات وتحضيرها بشكل صحيح
        # افتراض أن العمود الأول هو TransactionID والعمود الثاني يحتوي على المنتجات
        if len(df_raw.columns) >= 2:
            df_clean = df_raw.dropna()
            
            # تحويل سلسلة المنتجات إلى قائمة
            transactions = df_clean.iloc[:, 1].apply(lambda x: [item.strip() for item in str(x).split(',')]).tolist()
            
            st.write(f"✅ عدد المعاملات بعد التنظيف: {len(transactions)}")
            
            # تحويل لـ One-hot encoding
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

            # تطبيق Apriori مع معايير أكثر واقعية
            freq_items = apriori(df_encoded, min_support=0.2, use_colnames=True)
            st.subheader("📈 العناصر المتكررة:")
            st.dataframe(freq_items.sort_values('support', ascending=False))

            # استخراج القواعد
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)
            
            st.write("📋 عدد القواعد المستخرجة:", len(rules))

            if not rules.empty:
                st.subheader("📌 القواعد المكتشفة:")
                # تحسين عرض القواعد
                display_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
                display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                display_rules = display_rules.sort_values('lift', ascending=False)
                st.dataframe(display_rules)
            else:
                st.warning("⚠️ لم يتم العثور على قواعد بالمعايير المحددة. حاول تخفيض min_support أو min_threshold.")
        
        else:
            st.error("الملف يجب أن يحتوي على عمودين على الأقل: TransactionID والمنتجات")

    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {str(e)}")
        st.error("تأكد من تنسيق الملف. يجب أن يحتوي العمود الثاني على المنتجات مفصولة بفواصل")
else:
    st.info("📥 الرجاء رفع ملف Excel أو CSV لبدء التحليل.")