# ================= 颜色设置模块放置在图表区域 =================
# ===== 新增：图表颜色设置 =====
st.header("🎨 Chart Color Settings")

col1, col2 = st.columns(2)
with col1:
    pie_color_0 = st.color_picker("Color for 'No change'", "#99ff99")
    pie_color_1 = st.color_picker("Color for 'Warmer'", "#ff9999")
    pie_color_2 = st.color_picker("Color for 'Cooler'", "#66b3ff")

with col2:
    scatter_color = st.color_picker("Scatter point color", "#000000")
    vline_color_min = st.color_picker("Vertical line color for min temperature", "#0000FF")
    vline_color_max = st.color_picker("Vertical line color for max temperature", "#FF0000")
    
lr_color_0 = st.color_picker("Color for Thermal preference 0", "#0000FF")
lr_color_1 = st.color_picker("Color for Thermal preference 1", "#FF0000")
lr_color_2 = st.color_picker("Color for Thermal preference 2", "#008000")

# ================= 预测分析模块 =================
st.header("🔮 Predictive analysis")
selected_model = st.selectbox("Selecting a Predictive Model", list(models.keys()))

if st.button("Start forecasting"):
    try:
        model = models[selected_model]
        # 对输入数据进行归一化处理
        scaled_df = scaler.transform(df)
        
        with st.spinner("Predictions are in progress, please wait..."):
            predictions = model.predict(scaled_df)
            proba = model.predict_proba(scaled_df) if hasattr(model, "predict_proba") else None

        results_df = df.copy()
        results_df["Projected results"] = predictions
        comfort_mapping = {
            0: "No change",
            1: "Warmer",
            2: "Cooler"
        }
        results_df["Comfort Evaluation"] = results_df["Projected results"].map(comfort_mapping)

        with st.expander("📊 View detailed forecast results", expanded=True):
            def highlight_tp(val):
                colors = {0: '#e6ffe6', 1: '#ffe6e6', 2: '#e6f3ff'}
                return f'background-color: {colors.get(val, "")}'
            styled_df = results_df.style.applymap(highlight_tp, subset=["Projected results"])
            st.dataframe(styled_df, height=300)

        st.subheader("📈 Analyzing Charts")
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
                results_df["Indoor Air Temperature"],
                results_df["Projected results"],
                c=scatter_color,
                alpha=0.7
            )
            zero_projected_results = results_df[results_df["Projected results"] == 0]
            if not zero_projected_results.empty:
                min_temp_at_zero = zero_projected_results["Indoor Air Temperature"].min()
                max_temp_at_zero = zero_projected_results["Indoor Air Temperature"].max()
                plt.axvline(x=min_temp_at_zero, color=vline_color_min, linestyle=':', 
                            label=f'Min Temp at Zero ({min_temp_at_zero:.2f}°C)')
                plt.axvline(x=max_temp_at_zero, color=vline_color_max, linestyle=':', 
                            label=f'Max Temp at Zero ({max_temp_at_zero:.2f}°C)')
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
            
            # 绘制多项逻辑回归概率曲线
            temp_range_multi = np.linspace(results_df["Indoor Air Temperature"].min(),
                                           results_df["Indoor Air Temperature"].max(),
                                           1000).reshape(-1, 1)
            proba_multi = lr_multi.predict_proba(temp_range_multi)
            fig_multi, ax_multi = plt.subplots(figsize=(10, 6))
            ax_multi.plot(temp_range_multi, proba_multi[:, 0], label="Thermal preference 0", 
                          color=lr_color_0, linewidth=2)
            ax_multi.plot(temp_range_multi, proba_multi[:, 1], label="Thermal preference 1", 
                          color=lr_color_1, linewidth=2)
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

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("可能原因：\n1. 输入数据格式错误\n2. 模型文件缺失\n3. 特征列不匹配")
