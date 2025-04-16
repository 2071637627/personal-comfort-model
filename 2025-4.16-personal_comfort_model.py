import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io  # 用于图形下载
import os  # 新增路径处理

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'LightGBM': joblib.load('lgbm_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl')
}

scaler = joblib.load('minmax_scaler.pkl')  # 加载训练时保存的归一化器

# 页面配置
st.set_page_config(
    page_title="Thermal comfort prediction system",
    page_icon="🌡️",
    layout="wide"
)

# ================= 侧边栏输入模块 =================
with st.sidebar:
    st.header("⚙ Parameter Input Panel")

    # Subject's Personal Information
    st.subheader("Subject's Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        Sex = st.radio("Sex", ["Female (0)", "Male (1)", "Unknown (3)"], index=0)
    with col2:
        Age = st.selectbox(
            "Age_Category",
            ["18岁以下", "18-30岁", "31-40岁", "41-50岁", "51-60岁", "61岁以上"],
            index=1
        )
    
    col3, col4 = st.columns(2)
    with col3:
        Height = st.number_input("Height", 100, 250, 170)
    with col4:
        Weight = st.number_input("Weight", 30, 150, 65)

    # 第四层级：Subjective Thermal Comfort Information
    st.subheader("Subjective Thermal Comfort Information")
    Clothing_Insulation = st.number_input("Clothing Insulation", 0.0, 2.0, 1.0, 0.1)
    Metabolic_Rate = st.number_input("Metabolic Rate", 0.5, 4.0, 1.2, 0.1)

    # 第五层级：Indoor Physical Parameters
    st.subheader("Indoor Physical Parameters")
    input_mode = st.radio(
        "Input pattern", 
        ["Randomly generate (30)", "Randomly generate (50)", "Randomly generate (100)"]
    )
    min_temp = -10  # 假设最小户外温度为 0°C
    max_temp = 35  # 假设最大户外温度为 30°C
    
    st.info(f"Outdoor temperature generation range：{min_temp}°C ~ {max_temp}°C")

    # ===== 新增：图表颜色设置 =====
    st.sidebar.header("🎨 Chart Color Settings")
    with st.sidebar.expander("Pie Chart Colors", expanded=False):
        pie_color_0 = st.color_picker("Color for 'No change'", "#99ff99")
        pie_color_1 = st.color_picker("Color for 'Warmer'", "#ff9999")
        pie_color_2 = st.color_picker("Color for 'Cooler'", "#66b3ff")
    with st.sidebar.expander("Scatter Plot Colors", expanded=False):
        scatter_color = st.color_picker("Scatter point color", "#000000")
        vline_color_min = st.color_picker("Vertical line color for min temperature", "#0000FF")
        vline_color_max = st.color_picker("Vertical line color for max temperature", "#FF0000")
    with st.sidebar.expander("Logistic Regression Curve Colors", expanded=False):
        lr_color_0 = st.color_picker("Color for Thermal preference 0", "#0000FF")
        lr_color_1 = st.color_picker("Color for Thermal preference 1", "#FF0000")
        lr_color_2 = st.color_picker("Color for Thermal preference 2", "#008000")

# ================= 数据处理模块 =================
def categorize_age(age_str):
    """将年龄分类映射为数值编码"""
    age_mapping = {
        "18岁以下": 0,
        "18-30岁": 1,
        "31-40岁": 2,
        "41-50岁": 3,
        "51-60岁": 4,
        "61岁以上": 5
    }
    return age_mapping.get(age_str, 0)
    
def generate_data():
    """Generate data frames that are strictly consistent with the training features"""
    # 性别编码处理
    sex_code = 3  # 默认未知
    if "Female" in Sex:
        sex_code = 0
    elif "Male" in Sex:
        sex_code = 1

    # 构造基础特征
    base_features = {
        'Sex': sex_code,
        'Age_Category': categorize_age(Age),
        'Height': Height,
        'Weight': Weight,
        'Clothing Insulation': Clothing_Insulation,
        'Metabolic Rate': Metabolic_Rate
    }

    # 生成环境参数
    n_samples = int(input_mode.split("(")[1].replace(")", ""))
    np.random.seed(42)
    env_params = pd.DataFrame({
        'Indoor Air Temperature': np.clip(np.random.normal(25, 5, n_samples), 10, 40),
        'Indoor Relative Humidity': np.random.uniform(30, 80, n_samples),
        'Indoor Air Velocity': np.abs(np.random.normal(0, 0.5, n_samples)),
        'Mean Daily Outdoor Temperature': np.random.uniform(min_temp, max_temp, n_samples)
    }).round(1)

    # 合并特征并排序
    feature_order = [
        'Sex', 'Age_Category', 'Height', 'Weight',
        'Clothing Insulation', 'Metabolic Rate',
        'Indoor Air Temperature', 'Indoor Relative Humidity',
        'Indoor Air Velocity', 'Mean Daily Outdoor Temperature'
    ]
    
    df = pd.concat([pd.DataFrame([base_features]*n_samples).reset_index(drop=True), env_params], axis=1)
    df = df[feature_order]
    
    # 特征重命名
    df.columns = [
        'Column_0', 'Column_1', 'Column_2', 'Column_3',
        'Column_4', 'Column_5', 'Column_6', 'Column_7',
        'Column_8', 'Column_9'
    ]
    
    return df

# ================= 主界面显示模块 =================
st.title("🏢 Intelligent Prediction System for Building Thermal Comfort")
df = generate_data()

with st.expander("📥 Viewing Input Data", expanded=True):
    st.dataframe(df.style.format("{:.1f}"), height=300)
    st.download_button(
        label="Download input data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='input_data.csv'
    )

# ================= 预测分析模块 =================
st.header("🔮 Predictive analysis")
selected_model = st.selectbox("Selecting a Predictive Model", list(models.keys()))

if st.button("Start forecasting"):
    try:
        model = models[selected_model]
        scaled_df = scaler.transform(df)
        
        # XGBoost特殊处理
        if selected_model == 'XGBoost':
            dmatrix = xgb.DMatrix(scaled_df)
            raw_pred = model.predict(dmatrix)
            predictions = np.argmax(raw_pred, axis=1)
            proba = raw_pred
        else:
            predictions = model.predict(scaled_df)
            proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None

        # 结果处理
        results = df.copy()
        results['Prediction'] = predictions
        results['Comfort Level'] = results['Prediction'].map({
            0: "No change", 
            1: "Warmer", 
            2: "Cooler"
        })

        
        # 结果展示
        with st.expander("📊 Prediction Results"):
            st.dataframe(results.style.applymap(
                lambda x: 'background-color: #e6ffe6' if x == 0 else 
                         'background-color: #ffe6e6' if x == 1 else 
                         'background-color: #e6f3ff',
                subset=['Prediction']
            ))

        # 可视化
        st.subheader("📈 Analysis Visualizations")
        col1, col2 = st.columns(2)
        
        # =========== 图形1：预测结果分布饼图 ===========
        with col1:
            fig1 = plt.figure(figsize=(8, 6))
            results_df["Comfort Evaluation"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=[pie_color_0, pie_color_1, pie_color_2],
                startangle=90,
                textprops={"fontsize": 12}
            )
            plt.title("Distribution of forecast results", fontsize=14)
            plt.ylabel("", fontsize=12)
            st.pyplot(fig1)
            
            # 下载饼图
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png')
            buf1.seek(0)
            st.download_button(
                label="Download Pie Chart",
                data=buf1,
                file_name="pie_chart.png",
                mime="image/png"
            )

        # =========== 图形2：温度-舒适度散点图 ===========
        with col2:
            fig2 = plt.figure(figsize=(8, 8))
            plt.scatter(
                results_df["Indoor Air Temperature"],  # 对应 Indoor Air Temperature
                results_df["Projected results"],
                c=scatter_color,
                alpha=0.7
            )
            zero_projected_results = results_df[results_df["Projected results"] == 0]
            if not zero_projected_results.empty:
                min_temp_at_zero = zero_projected_results["Indoor Air Temperature"].min()
                max_temp_at_zero = zero_projected_results["Indoor Air Temperature"].max()
                #plt.axvline(x=min_temp_at_zero, color=vline_color_min, linestyle=':', 
                            #label=f'Min Temp at Zero ({min_temp_at_zero:.2f}°C)')
                #plt.axvline(x=max_temp_at_zero, color=vline_color_max, linestyle=':', 
                            #label=f'Max Temp at Zero ({max_temp_at_zero:.2f}°C)')
            plt.legend()
            plt.title("Mapping of indoor air temperatures to predicted thermal preferences", fontsize=14)
            plt.xlabel("Indoor Air Temperature", fontsize=12)
            plt.ylabel("Thermal preference", fontsize=12)
            plt.grid(linestyle="--", alpha=0.3)
            st.pyplot(fig2)
            
            # 下载散点图
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png')
            buf2.seek(0)
            st.download_button(
                label="Download Scatter Plot",
                data=buf2,
                file_name="scatter_plot.png",
                mime="image/png"
            )

        # 下载预测结果数据
        st.download_button(
            label="Download full forecast results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name=f'predictions_{selected_model}.csv',
            mime='text/csv'
        )

        # ----------------- 新增：多项逻辑回归曲线及参数显示 -----------------
        with st.expander("📈 Multinomial Logistic Regression Curves", expanded=True):
    # 检查是否为手动输入模式且数据量是否大于等于10
            if "Manual" in input_mode and len(df) >= 10 or "Randomly" in input_mode:
        # 使用“Indoor Air Temperature”作为唯一特征构造多项逻辑回归模型
                X_multi = results_df["Indoor Air Temperature"].values.reshape(-1, 1)
                y_multi = results_df["Projected results"].values
                lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
                lr_multi.fit(X_multi, y_multi)

                # 显示每一条回归曲线的参数和回归公式
                st.markdown("### Regression curve parameters and regression equation")
                intercepts = lr_multi.intercept_
                coefs = lr_multi.coef_  # shape (3, 1)

        # 遍历 3 个类别
                for idx in range(len(intercepts)):
                    intercept = intercepts[idx]
                    coef = coefs[idx][0]
                    st.markdown(f"**Thermal preference {idx} （{comfort_mapping[idx]}）**")
                    st.write(f"Intercept (β₀): {intercept:.4f}")
                    st.write(f"Temperature coefficient (β₁): {coef:.4f}")
                    st.markdown(
                f"**Regression equation:** $$p_{{{idx}}}(x)=\\frac{{\\exp({intercept:.4f}+{coef:.4f}x)}}{{\\exp({intercepts[0]:.4f}+{coefs[0][0]:.4f}x)+\\exp({intercepts[1]:.4f}+{coefs[1][0]:.4f}x)+\\exp({intercepts[2]:.4f}+{coefs[2][0]:.4f}x)}}$$"
                    )

        # 选择是否显示每条逻辑回归曲线
                #show_lr_0 = st.checkbox("Show Thermal preference 0 curve", value=True)
                #show_lr_1 = st.checkbox("Show Thermal preference 1 curve", value=True)
                #show_lr_2 = st.checkbox("Show Thermal preference 2 curve", value=True)

        # 绘制多项逻辑回归概率曲线
                temp_range_multi = np.linspace(results_df["Indoor Air Temperature"].min(),
                                               results_df["Indoor Air Temperature"].max(),
                                               1000).reshape(-1, 1)
                proba_multi = lr_multi.predict_proba(temp_range_multi)

                fig_multi, ax_multi = plt.subplots(figsize=(10, 6))

                #if show_lr_0:
                ax_multi.plot(temp_range_multi, proba_multi[:, 0], label="Thermal preference 0",
                              color=lr_color_0, linewidth=2)
                #if show_lr_1:
                ax_multi.plot(temp_range_multi, proba_multi[:, 1], label="Thermal preference 1",
                              color=lr_color_1, linewidth=2)
                #if show_lr_2:
                ax_multi.plot(temp_range_multi, proba_multi[:, 2], label="Thermal preference 2",
                              color=lr_color_2, linewidth=2)

                ax_multi.set_xlabel("Indoor Air Temperature (°C)", fontsize=12)
                ax_multi.set_ylabel("Predicted Probability", fontsize=12)
                ax_multi.set_title("Multinomial Logistic Regression Curves for Thermal Preference", fontsize=14)
                ax_multi.legend()
                ax_multi.grid(linestyle="--", alpha=0.3)
                st.pyplot(fig_multi)

        # 下载逻辑回归曲线图
                buf3 = io.BytesIO()
                fig_multi.savefig(buf3, format='png')
                buf3.seek(0)
                st.download_button(
                    label="Download Logistic Regression Curve",
                    data=buf3,
                    file_name="logistic_regression_curve.png",
                    mime="image/png"
                )
            else:
                st.warning("The amount of data is too small to generate a multinomial logistic regression curve. Please manually enter at least 10 data points.")

    except Exception as e:
        st.error(f"Prediction Failure:{str(e)}")
        st.error("Possible causes: \n1. input data format error \n2. model file missing \n3. feature column mismatch")