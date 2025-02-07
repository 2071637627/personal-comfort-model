import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'LightGBM': joblib.load('lgbm_model.pkl'),
    'GBM': joblib.load('gbm_model.pkl'),
    'XGBoost': joblib.load('xgb_model.pkl')
}

# 加载标准化器
scaler = joblib.load('scaler.pkl')  # 确保与训练时使用的scaler一致

# 页面配置
st.set_page_config(
    page_title="热舒适度预测系统",
    page_icon="🌡️",
    layout="wide"
)

# ================= 侧边栏输入模块 =================
with st.sidebar:
    st.header("⚙️ 参数输入面板")

    # 第一层级：Basic Identifiers
    st.subheader("1. 基础标识")
    season = st.selectbox(
        "季节",
        ["冬季 (0)", "夏季 (1)", "过渡季 (2)"],
        index=0
    )
    climate_zone = st.selectbox(
        "气候分区",
        ["严寒地区 (0)", "寒冷地区 (1)", "夏热冬冷 (2)",
         "夏热冬暖 (3)", "温和地区 (4)"],
        index=0
    )

    # 第二层级：Building Information
    st.subheader("2. 建筑信息")
    building_type = st.selectbox(
        "建筑类型",
        ["宿舍 (0)", "教育建筑 (1)", "办公建筑 (2)", "住宅 (3)"],
        index=0
    )
    operation_mode = st.selectbox(
        "运行模式",
        ["空调供暖 (0)", "毛细管顶棚供暖 (1)", 
         "冷辐射顶棚供冷 (2)", "对流供冷 (3)",
         "对流供暖 (4)", "锅炉供暖 (5)", 
         "自然通风 (6)", "其他 (7)",
         "地板辐射供暖 (8)", "散热器供暖 (9)",
         "自采暖 (10)", "分体空调 (11)"],
        index=0
    )

    # 第三层级：Subject's Personal Information
    st.subheader("3. 人员信息")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.radio("性别", ["女 (0)", "男 (1)"], index=0)
    with col2:
        age_group = st.selectbox(
            "年龄段",
            ["<18 (0)", "18-30 (1)", "31-40 (2)", 
             "41-50 (3)", "51-60 (4)", ">61 (5)"],
            index=1
        )
    
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("身高 (cm)", 100, 250, 170)
    with col4:
        weight = st.number_input("体重 (kg)", 30, 150, 65)

    # 第四层级：Subjective Thermal Comfort Information
    st.subheader("4. 热舒适参数")
    clothing = st.number_input("服装热阻 (clo)", 0.0, 2.0, 1.0, 0.1)
    metabolic = st.number_input("代谢率 (met)", 0.5, 4.0, 1.2, 0.1)

    # 第五层级：Indoor Physical Parameters
    st.subheader("5. 室内外环境参数")
    input_mode = st.radio(
        "输入模式", 
        ["手动输入", "随机生成30组", "随机生成50组", "随机生成100组"]
    )

    # 气候分区温度范围
    climate_code = int(climate_zone.split("(")[1].replace(")", ""))
    temp_ranges = {
        0: (-20, 5),    # 严寒地区
        1: (-10, 10),   # 寒冷地区
        2: (0, 25),     # 夏热冬冷
        3: (10, 35),    # 夏热冬暖
        4: (5, 30)      # 温和地区
    }
    min_temp, max_temp = temp_ranges[climate_code]

    if "手动" in input_mode:
        # 自动计算合理默认值
        default_temp = np.clip(15.0, min_temp, max_temp)  # 确保默认值在有效范围内
        
        outdoor_temp = st.number_input(
            "日均室外温度 (°C)",
            min_value=float(min_temp),  # 确保转换为float
            max_value=float(max_temp),
            value=default_temp,
            step=0.5,
            format="%.1f",
            help=f"当前气候分区有效范围：{min_temp}°C ~ {max_temp}°C"
        )
    else:
        st.info(f"室外温度生成范围：{min_temp}°C ~ {max_temp}°C")

# ================= 数据处理模块 =================
def generate_data():
    """生成输入数据框"""
    # 解析编码值
    codes = {
        'Season': int(season.split("(")[1].replace(")", "")),
        'Climate Zone': int(climate_zone.split("(")[1].replace(")", "")),
        'Building Type': int(building_type.split("(")[1].replace(")", "")),
        'Operation Mode': int(operation_mode.split("(")[1].replace(")", "")),
        'Sex': int(sex.split("(")[1].replace(")", "")),
        'Age Group': int(age_group.split("(")[1].replace(")", "")),
        'Height (cm)': height,
        'Weight (kg)': weight,
        'Clothing (clo)': clothing,
        'Metabolic (met)': metabolic
    }

    # 生成环境参数
    if "手动" in input_mode:
        temp = st.number_input("空气温度 (°C)", 10.0, 40.0, 25.0)
        humidity = st.number_input("相对湿度 (%)", 0.0, 100.0, 50.0)
        velocity = st.number_input("空气流速 (m/s)", 0.0, 5.0, 0.1)
        env_params = [[temp, humidity, velocity, outdoor_temp]]  # 添加室外温度
    else:
        n_samples = int(input_mode.split("生成")[1].replace("组", ""))
        np.random.seed(42)
        env_params = np.column_stack([
            np.round(np.random.uniform(18, 32, n_samples), 1),
            np.round(np.random.uniform(30, 80, n_samples), 1),
            np.round(np.random.uniform(0, 1.5, n_samples), 2),
            np.round(np.random.uniform(min_temp, max_temp, n_samples), 1)  # 添加室外温度
        ])

    # 构建数据框
    df = pd.DataFrame(env_params, columns=[
        'Temperature (°C)', 'Humidity (%)', 'Velocity (m/s)', 'Outdoor Temperature (°C)'
    ])
    
    # 添加固定参数
    for col, val in codes.items():
        df[col] = val

    # 调整列顺序
    feature_order = [
        'Season', 'Climate Zone', 'Building Type', 'Operation Mode',
        'Sex', 'Age Group', 'Height (cm)', 'Weight (kg)',
        'Clothing (clo)', 'Metabolic (met)',
        'Temperature (°C)', 'Humidity (%)', 'Velocity (m/s)', 'Outdoor Temperature (°C)'
    ]
    return df[feature_order]

# ================= 主界面显示模块 =================
st.title("🏢 建筑热舒适度智能预测系统")
df = generate_data()

# 输入数据展示
with st.expander("📥 查看输入数据", expanded=True):
    st.dataframe(df.style.format("{:.1f}"), height=300)
    st.download_button(
        label="下载输入数据",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='input_data.csv'
    )

# ================= 预测分析模块 =================
st.header("🔮 预测分析")
selected_model = st.selectbox("选择预测模型", list(models.keys()))

if st.button("开始预测"):
    try:
        model = models[selected_model]
        
        # 对输入数据进行归一化处理
        scaled_df = scaler.transform(df)  # 使用标准化器对数据进行归一化
        scaled_df = pd.DataFrame(scaled_df, columns=df.columns)  # 将归一化后的数据转换回DataFrame

        # 执行预测
        with st.spinner("预测进行中，请稍候..."):
            predictions = model.predict(scaled_df)  # 使用归一化后的数据进行预测
            proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None

        # 构建结果数据框
        results_df = df.copy()  # 使用原始数据框作为基础
        results_df["预测结果"] = predictions
        # 定义舒适度评价的映射关系
        comfort_mapping = {
            0: "无需改变",
            1: "希望更暖",
            2: "希望更凉"
        }
        # 使用 map 函数将预测结果映射为舒适度评价
        results_df["舒适度评价"] = results_df["预测结果"].map(comfort_mapping)

        # 显示预测结果
        with st.expander("📊 查看详细预测结果", expanded=True):
            # 条件格式
            def highlight_tp(val):
                colors = {0: '#e6f3ff', 1: '#ffe6e6', 2: '#e6ffe6'}
                return f'background-color: {colors.get(val, "")}'
            
            styled_df = results_df.style.applymap(highlight_tp, subset=["预测结果"])
            st.dataframe(styled_df, height=300)

        # 可视化分析
        st.subheader("📈 分析图表")
        col1, col2 = st.columns(2)

        with col1:
            # 预测结果分布
            fig1 = plt.figure(figsize=(8, 6))
            results_df["舒适度评价"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=["#66b3ff", "#ff9999", "#99ff99"],
                startangle=90
            )
            plt.title("预测结果分布")
            plt.ylabel("")
            st.pyplot(fig1)

        with col2:
            # 温度-舒适度关系
            fig2 = plt.figure(figsize=(8, 6))
            plt.scatter(
                results_df["Temperature (°C)"],
                results_df["预测结果"],
                c=results_df["预测结果"],
                cmap="coolwarm",
                alpha=0.7
            )
            plt.colorbar(ticks=[0, 1, 2]).set_ticklabels(["无需改变", "希望更暖", "希望更凉"])
            plt.xlabel("温度 (°C)")
            plt.ylabel("热舒适偏好")
            plt.grid(linestyle="--", alpha=0.3)
            st.pyplot(fig2)

        # 下载结果
        st.download_button(
            label="下载完整预测结果",
            data=results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'predictions_{selected_model}.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("可能原因：\n1. 输入数据格式错误\n2. 模型文件缺失\n3. 特征列不匹配")