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

# 页面配置
st.set_page_config(
    page_title="智能热舒适预测系统",
    page_icon="🌡️",
    layout="wide"
)

# ================= 侧边栏输入模块 =================
with st.sidebar:
    st.header("⚙️ 参数输入面板")
    
    # 1. 环境基本信息
    st.subheader("1. 环境基本信息")
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
    
    # 2. 建筑信息
    st.subheader("2. 建筑信息")
    building_type = st.selectbox(
        "建筑类型",
        ["宿舍 (0)", "教育建筑 (1)", "办公建筑 (2)", "住宅 (3)"],
        index=0
    )
    operation_mode = st.selectbox(
        "运行模式",
        ["空调供暖 (0)", "毛细管顶棚供暖 (1)", "冷辐射顶棚供冷 (2)",
         "对流供冷 (3)", "对流供暖 (4)", "锅炉供暖 (5)", 
         "自然通风 (6)", "其他 (7)", "地板辐射供暖 (8)",
         "散热器供暖 (9)", "自采暖 (10)", "分体空调 (11)"],
        index=0
    )
    
    # 3. 人员信息
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
    
    # 4. 热舒适参数
    st.subheader("4. 热舒适参数")
    clothing = st.number_input("服装热阻 (clo)", 0.0, 2.0, 1.0, 0.1)
    metabolic = st.number_input("代谢率 (met)", 0.5, 4.0, 1.2, 0.1)
    
    # 5. 室内外环境参数
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

# ================= 数据生成模块 =================
def generate_data():
    """生成包含室外温度的数据框"""
    # 解析编码值
    codes = {
        'Season': int(season.split("(")[1].replace(")", "")),
        'Climate Zone': climate_code,
        'Building Type': int(building_type.split("(")[1].replace(")", "")),
        'Operation Mode': int(operation_mode.split("(")[1].replace(")", "")),
        'Sex': int(sex.split("(")[1].replace(")", "")),
        'Age Group': int(age_group.split("(")[1].replace(")", "")),
        'Height (cm)': height,
        'Weight (kg)': weight,
        'Clothing (clo)': clothing,
        'Metabolic (met)': metabolic,
        'BMI': weight / ((height/100)**2)
    }
    
    # 生成环境参数
    if "手动" in input_mode:
        indoor_temp = st.number_input("室内温度 (°C)", 10.0, 40.0, 25.0)
        humidity = st.number_input("相对湿度 (%)", 0.0, 100.0, 50.0)
        velocity = st.number_input("空气流速 (m/s)", 0.0, 5.0, 0.1)
        env_params = [[indoor_temp, humidity, velocity, outdoor_temp]]
    else:
        n_samples = int(input_mode.split("生成")[1].replace("组", ""))
        np.random.seed(42)
        
        indoor_temp = np.round(np.random.uniform(18, 32, n_samples), 1)
        humidity = np.round(np.random.uniform(30, 80, n_samples), 1)
        velocity = np.round(np.random.uniform(0, 1.5, n_samples), 2)
        outdoor_temp = np.round(np.random.uniform(min_temp, max_temp, n_samples), 1)
        
        env_params = np.column_stack([indoor_temp, humidity, velocity, outdoor_temp])
    
    # 构建数据框
    df = pd.DataFrame(
        env_params,
        columns=[
            'Temperature (°C)', 
            'Humidity (%)', 
            'Velocity (m/s)',
            'Outdoor Temp (°C)'
        ]
    )
    
    # 添加其他参数
    for col, val in codes.items():
        df[col] = val
    
    # 调整特征顺序
    feature_order = [
        'Season', 'Climate Zone', 'Building Type', 'Operation Mode',
        'Sex', 'Age Group', 'Height (cm)', 'Weight (kg)',
        'Clothing (clo)', 'Metabolic (met)', 'BMI',
        'Temperature (°C)', 'Humidity (%)', 
        'Velocity (m/s)', 'Outdoor Temp (°C)'
    ]
    return df[feature_order]

# ================= 主界面模块 =================
st.title("🏢 建筑热舒适度智能预测系统")
df = generate_data()

# 输入数据展示
with st.expander("📥 查看输入数据", expanded=True):
    st.dataframe(df.style.format("{:.1f}"), height=300)
    st.download_button(
        label="下载输入数据",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='thermal_comfort_input.csv'
    )

# ================= 预测分析模块 =================
st.header("🔮 预测分析")
selected_model = st.selectbox("选择预测模型", list(models.keys()))

if st.button("开始预测"):
    try:
        model = models[selected_model]
        
        # 特征验证
        required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        if len(df.columns) != len(required_features):
            st.error("特征数量不匹配！请检查：")
            st.write(f"模型需要 {len(required_features)} 个特征，当前 {len(df.columns)} 个")
            st.write("模型特征列表:", required_features)
            st.write("当前特征列表:", df.columns.tolist())
            st.stop()
        
        # 执行预测
        with st.spinner("预测进行中..."):
            predictions = model.predict(df)
            results_df = df.copy()
            results_df["预测结果"] = predictions
            results_df["舒适度评价"] = results_df["预测结果"].map({
                0: "无需改变",
                1: "希望更暖",
                2: "希望更凉"
            })
        
        # 显示结果
        with st.expander("📊 查看详细预测结果", expanded=True):
            st.dataframe(
                results_df.style.applymap(
                    lambda x: "#e6f3ff" if x==0 else "#ffe6e6" if x==1 else "#e6ffe6",
                    subset=["预测结果"]
                ),
                height=400
            )
        
        # 可视化分析
        st.subheader("📈 分析图表")
        
        # 结果分布
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plt.figure(figsize=(8,6))
            results_df["舒适度评价"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=["#66b3ff","#ff9999","#99ff99"],
                startangle=90
            )
            plt.title("预测结果分布")
            st.pyplot(fig1)
        
        # 温度分布
        with col2:
            fig2 = plt.figure(figsize=(8,6))
            plt.hist(results_df["Temperature (°C)"], bins=15, edgecolor="k")
            plt.xlabel("室内温度 (°C)")
            plt.ylabel("频次")
            plt.title("温度分布直方图")
            st.pyplot(fig2)
        
        # 室内外温度关系
        st.subheader("🌍 室内外温度分析")
        fig3 = plt.figure(figsize=(10,6))
        plt.scatter(
            results_df["Outdoor Temp (°C)"],
            results_df["Temperature (°C)"],
            c=results_df["预测结果"],
            cmap="coolwarm",
            alpha=0.7
        )
        plt.colorbar(label="热舒适偏好", ticks=[0,1,2]).set_ticklabels(["无需改变","希望更暖","希望更凉"])
        plt.xlabel("室外温度 (°C)")
        plt.ylabel("室内温度 (°C)")
        plt.grid(linestyle="--", alpha=0.3)
        st.pyplot(fig3)
        
        # 下载结果
        st.download_button(
            label="下载完整预测结果",
            data=results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'thermal_comfort_predictions_{selected_model}.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("常见原因：\n1. 特征数量/顺序不匹配\n2. 模型文件损坏\n3. 输入数据超出范围")

# ================= 帮助信息 =================
with st.expander("❓ 使用帮助"):
    st.markdown("""
    **使用指南：**
    1. 在左侧面板输入所有参数
    2. 选择数据生成模式（手动/随机）
    3. 选择预测模型
    4. 点击「开始预测」查看结果
    
    **特征说明：**
    - BMI = 体重(kg) / (身高(m)^2)
    - 室外温度范围根据气候分区自动调整
    - 随机生成数据符合ASHRAE标准范围
    
    **技术支持：**
    - 模型更新日期：2023-12-01
    - 数据版本：v2.1.5
    """)