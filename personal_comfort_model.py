import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载模型和标准化器
model_xgb = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# 创建映射字典
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
    '[18...30]': 1,
    '[31...50]': 2,
    '[51...]': 3
}

# Streamlit界面
st.title("Building Energy Prediction")

# 获取用户输入
season = st.selectbox("Select Season", options=list(season_map.keys()))
climate_zone = st.selectbox("Select Climate Zone", options=list(climate_zone_map.keys()))
building_type = st.selectbox("Select Building Type", options=list(building_type_map.keys()))
operation_mode = st.selectbox("Select Operation Mode", options=list(operation_mode_map.keys()))
sex = st.selectbox("Select Sex", options=list(sex_map.keys()))
age = st.selectbox("Select Age Range", options=list(age_map.keys()))

# 将输入转换为数值
season_value = season_map[season]
climate_zone_value = climate_zone_map[climate_zone]
building_type_value = building_type_map[building_type]
operation_mode_value = operation_mode_map[operation_mode]
sex_value = sex_map[sex]
age_value = age_map[age]

# 假设这些是其他需要的输入（你可以根据实际情况添加更多输入）
temperature = st.slider("Select Temperature", min_value=-10, max_value=40, value=20)
humidity = st.slider("Select Humidity", min_value=0, max_value=100, value=50)

# 创建数据框架
input_data = pd.DataFrame({
    'season': [season_value],
    'climate_zone': [climate_zone_value],
    'building_type': [building_type_value],
    'operation_mode': [operation_mode_value],
    'sex': [sex_value],
    'age': [age_value],
    'temperature': [temperature],
    'humidity': [humidity]
})

# 对输入数据进行标准化
input_scaled = scaler.transform(input_data)

# 进行预测
prediction = model_xgb.predict(input_scaled)

# 显示预测结果
st.write("Predicted Energy Consumption:", prediction[0])
