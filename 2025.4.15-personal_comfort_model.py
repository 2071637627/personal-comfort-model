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
    'LightGBM': joblib.load('lgbm_model.pkl')
    #'XGBoost': joblib.load('xgb_model.pkl')
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
        Sex = st.radio("Sex", ["Female (0)", "Male (1)"], index=0)
    with col2:
        Age = st.selectbox(
            "Age_Category",
            ["<18 (0)", "18-30 (1)", "31-40 (2)", "41-50 (3)", "51-60 (4)", ">61 (5)"],
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
def generate_data():
    """生成与模型特征严格匹配的数据框"""
    # 解析输入参数
    sex_code = int(Sex.split("(")[1].replace(")", ""))
    age_code = int(Age.split("(")[1].replace(")", ""))
    n_samples = int(input_mode.split("(")[1].replace(")", ""))

    # 静态参数（转换为列表）
    static_features = {
        'Sex': [sex_code] * n_samples,
        'Age_Category': [age_code] * n_samples,
        'Height': [Height] * n_samples,
        'Weight': [Weight] * n_samples,
        'Clothing Insulation': [Clothing_Insulation] * n_samples,
        'Metabolic Rate': [Metabolic_Rate] * n_samples
    }

    # 环境参数（随机生成）
    np.random.seed(42)
    env_features = {
        'Indoor Air Temperature': np.round(np.random.uniform(10, 40, n_samples), 1),
        'Indoor Relative Humidity': np.round(np.random.uniform(30, 80, n_samples), 1),
        'Indoor Air Velocity': np.round(np.random.uniform(0, 1.5, n_samples), 2),
        'Mean Daily Outdoor Temperature': np.round(np.random.uniform(min_temp, max_temp, n_samples), 1)
    }

    # 合并数据
    df = pd.DataFrame({**static_features, **env_features})

    # 特征顺序验证
    feature_order = [
        'Sex',
        'Age_Category',
        'Height',
        'Weight',
        'Clothing Insulation',
        'Metabolic Rate',
        'Indoor Air Temperature',
        'Indoor Relative Humidity',
        'Indoor Air Velocity',
        'Mean Daily Outdoor Temperature'
    ]
    assert set(df.columns) == set(feature_order), \
        f"列缺失/多余：应有 {feature_order}，实际 {df.columns.tolist()}"

    # 特征名称映射
    feature_mapping = {
        'Sex': 'Column_0',
        'Height': 'Column_1',
        'Weight': 'Column_2',
        'Clothing Insulation': 'Column_3',
        'Metabolic Rate': 'Column_4',
        'Indoor Air Temperature': 'Column_5',
        'Indoor Relative Humidity': 'Column_6',
        'Indoor Air Velocity': 'Column_7',
        'Mean Daily Outdoor Temperature': 'Column_8',
        'Age_Category': 'Column_9'
    }
    df = df.rename(columns=feature_mapping)[
        ['Column_0', 'Column_1', 'Column_2', 'Column_3',
         'Column_4', 'Column_5', 'Column_6', 'Column_7',
         'Column_8', 'Column_9']
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
        # 数据验证
        expected_columns = [
            'Column_0', 'Column_1', 'Column_2', 'Column_3',
            'Column_4', 'Column_5', 'Column_6', 'Column_7',
            'Column_8', 'Column_9'
        ]
        assert df.columns.tolist() == expected_columns, \
            f"特征列不匹配！当前列：{df.columns.tolist()}"
        
        # 对输入数据进行归一化处理
        scaled_data = scaler.transform(df)
        
        # 模型预测
        model = models[selected_model]
        predictions = model.predict(scaled_data)
        proba = model.predict_proba(scaled_data) if hasattr(model, "predict_proba") else None

        # 结果处理
        results = pd.DataFrame({
            'Indoor Temp': df['Column_5'],
            'Prediction': predictions,
            'Probability_0': proba[:,0] if proba is not None else None,
            'Probability_1': proba[:,1] if proba is not None else None,
            'Probability_2': proba[:,2] if proba is not None else None
        })
        results['Comfort Level'] = results['Prediction'].map({
            0: "No change",
            1: "Warmer",
            2: "Cooler"
        })

        # 结果可视化
        st.subheader("📊 Prediction Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8,6))
            results['Comfort Level'].value_counts().plot.pie(
                autopct='%1.1f%%', colors=pie_colors, ax=ax1
            )
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.scatter(
                results['Indoor Temp'],
                results['Prediction'],
                c=scatter_color,
                alpha=0.7
            )
            ax2.set_xlabel("Indoor Temperature (°C)")
            ax2.set_ylabel("Thermal Preference")
            st.pyplot(fig2)

        # 下载结果
        st.download_button(
            label="Download Full Results",
            data=results.to_csv(index=False).encode('utf-8'),
            file_name='prediction_results.csv'
        )

    except Exception as e:
        st.error(f"Prediction Failed: {str(e)}")
        st.error("""
        **常见故障排除步骤**：
        1. 检查所有输入参数是否在合理范围内
        2. 验证模型文件与代码版本是否匹配
        3. 确认特征列顺序与训练时一致
        """)

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