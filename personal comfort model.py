import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型和标准化器
model_xgb = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# 创建映射字典（与原始代码相同）
season_map = {'Winter': 0, 'Summer': 1, 'Transition': 2}
climate_zone_map = {
    'Sever cold zone': 0,
    'Cold zone': 1,
    'Hot summer and cold winter zone': 2,
    'Hot summer and warm winter zone': 3,
    'Mild zones': 4
}
building_type_map = {
    'Dormitory': 0,
    'Educational': 1,
    'Office': 2,
    'Residential': 3
}
operation_mode_map = {
    'Air conditioning heating': 0,
    'Ceiling capillary heating': 1,
    'Cold radiation ceiling cooling': 2,
    'Convection cooling': 3,
    'Convection heating': 4,
    'Furnace heating': 5,
    'Naturally Ventilated': 6,
    'Radiant floor heating': 7,
    'Radiator heating': 8,
    'self-heating': 9,
    'Split air conditioner': 10,
    'Others': 11
}
sex_map = {'female': 0, 'male': 1}
age_map = {
    '<18': 0,
    '[18, 30]': 1,
    '[31, 40]': 2,
    '[41, 50]': 3,
    '[51, 60]': 4,
    '>61': 5
}

# 网页标题
st.title("热舒适预测模型")
st.markdown("请输入以下特征数据：")

# 创建输入表单
with st.form("input_form"):
    # 分类特征的下拉菜单
    season = st.selectbox("Season", options=list(season_map.keys()))
    climate_zone = st.selectbox("Climate Zone", options=list(climate_zone_map.keys()))
    building_type = st.selectbox("Building Type", options=list(building_type_map.keys()))
    operation_mode = st.selectbox("Building Operation Mode", options=list(operation_mode_map.keys()))
    sex = st.selectbox("Sex", options=list(sex_map.keys()))
    age_group = st.selectbox("Age Group", options=list(age_map.keys()))
    
    # 数值特征的输入
    height = st.number_input("Height (cm)", min_value=0.0)
    weight = st.number_input("Weight (kg)", min_value=0.0)
    clothing = st.number_input("Clothing Insulation", min_value=0.0)
    metabolic = st.number_input("Metabolic Rate", min_value=0.0)
    indoor_temp = st.number_input("Indoor Air Temperature (°C)")
    indoor_humidity = st.number_input("Indoor Relative Humidity (%)", min_value=0.0, max_value=100.0)
    indoor_air_velocity = st.number_input("Indoor Air Velocity (m/s)", min_value=0.0)
    outdoor_temp = st.number_input("Mean Daily Outdoor Temperature (°C)")
    
    submitted = st.form_submit_button("预测")

if submitted:
    try:
        # 将分类特征转换为编码
        features = [
            season_map[season],
            climate_zone_map[climate_zone],
            building_type_map[building_type],
            operation_mode_map[operation_mode],
            sex_map[sex],
            age_map[age_group],
            float(height),
            float(weight),
            float(clothing),
            float(metabolic),
            float(indoor_temp),
            float(indoor_humidity),
            float(indoor_air_velocity),
            float(outdoor_temp)
        ]
        
        # 创建DataFrame
        feature_names = [
            'Season', 'Climate Zone', 'Building Type', 'Building Operation Mode', 'Sex', 
            'Age', 'Height', 'Weight', 'Clothing Insulation', 'Metabolic Rate', 
            'Indoor Air Temperature', 'Indoor Relative Humidity', 'Indoor Air Velocity', 
            'Mean Daily Outdoor Temperature'
        ]
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # 标准化处理
        input_scaled = scaler.transform(input_df)
        
        # 进行预测
        prediction = model_xgb.predict(input_scaled)
        
        # 显示结果
        st.success(f"预测的TP目标变量值为: {prediction[0]:.2f}")
        
    except Exception as e:
        st.error(f"输入错误: {str(e)}")

# 侧边栏说明
st.sidebar.markdown("""
### 使用说明
1. 填写/选择所有输入参数
2. 点击「预测」按钮
3. 查看底部预测结果

**注意：**
- 分类参数请从下拉菜单中选择
- 数值参数请输入有效数值
""")