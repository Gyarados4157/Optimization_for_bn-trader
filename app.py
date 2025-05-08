import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skfolio import Population, RiskMeasure
from skfolio.optimization import (
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
    EqualWeighted,
    HierarchicalEqualRiskContribution
)
from skfolio.preprocessing import prices_to_returns
from skfolio.exceptions import OptimizationError

st.set_page_config(page_title="BN 带单员保证金比例优化", layout="wide")
st.title("BN 带单员保证金比例自动配置系统")

# 本地文件路径输入
file_path = st.sidebar.text_input(
    "本地 JSONL 文件路径",
    value="portfolio_daily_stats.jsonl"
)

@st.cache_data
def load_and_process_jsonl(path: str) -> pd.DataFrame:
    portfolios_raw = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    portfolios_raw.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass # Reduce warnings
                    continue
    except FileNotFoundError:
        st.error(f"错误：文件 '{path}' 未找到。请检查路径是否正确。")
        return pd.DataFrame()
    if not portfolios_raw:
        st.error("JSONL 文件为空或所有行都无法解析。")
        return pd.DataFrame()
    valid_portfolios_for_metadata = [
        p for p in portfolios_raw
        if (p and isinstance(p.get('data'), dict) and
            p['data'].get('startDate') and p['data'].get('endDate') and
            isinstance(p['data'].get('margin'), list) and p['data']['margin'])
    ]
    if not valid_portfolios_for_metadata:
        st.error("未能从数据中找到包含有效 'startDate', 'endDate', 和非空 'margin' 列表的记录。")
        return pd.DataFrame()
    max_length = 0
    global_start_date, global_end_date = None, None
    for p_meta in valid_portfolios_for_metadata:
        max_length = max(max_length, len(p_meta['data']['margin']))
        try:
            s_date = datetime.fromisoformat(p_meta['data']['startDate'].replace('Z', '+00:00'))
            e_date = datetime.fromisoformat(p_meta['data']['endDate'].replace('Z', '+00:00'))
            if global_start_date is None or s_date < global_start_date: global_start_date = s_date
            if global_end_date is None or e_date > global_end_date: global_end_date = e_date
        except ValueError: continue
    if max_length == 0 or global_start_date is None or global_end_date is None:
        st.error("无法确定有效的全局最大长度或日期范围。")
        return pd.DataFrame()
    idx = pd.date_range(start=global_start_date, periods=max_length, freq='D')
    processed_data = {}
    for p in portfolios_raw:
        nickname = p.get('nickname')
        if not nickname: continue
        portfolio_data = p.get('data')
        if not isinstance(portfolio_data, dict): continue
        margins_str = portfolio_data.get('margin')
        if not isinstance(margins_str, list): continue
        try:
            margins_float = [float(x) for x in margins_str]
        except ValueError: continue
        current_len = len(margins_float)
        if current_len == 0: padded_margins = [0.0] * max_length
        elif current_len < max_length: padded_margins = margins_float + [margins_float[-1]] * (max_length - current_len)
        elif current_len > max_length: padded_margins = margins_float[:max_length]
        else: padded_margins = margins_float
        processed_data[nickname] = padded_margins
    if not processed_data:
        st.error("处理后没有有效的带单员数据。")
        return pd.DataFrame()
    final_df = pd.DataFrame(processed_data, index=idx)
    final_df.dropna(axis=1, how='all', inplace=True)
    return final_df

df_prices = load_and_process_jsonl(file_path)
if df_prices.empty:
    st.warning("未能加载或处理数据，应用程序已停止。")
    st.stop()
st.success(f"成功加载并处理了 {len(df_prices.columns)} 位带单员的数据。")
st.info(f"数据时间范围: {df_prices.index.min().strftime('%Y-%m-%d')} to {df_prices.index.max().strftime('%Y-%m-%d')}")

st.sidebar.header("配置选项")
available_traders = df_prices.columns.tolist()
if not available_traders:
    st.sidebar.error("数据文件中没有可用的带单员。")
    st.stop()

# --- Budget Selection ---
budget_type = st.sidebar.radio(
    "选择预算类型 (Budget Type)",
    options=["固定总和 (Fixed Sum)", "无总和限制 (No Sum Constraint)"],
    index=0,
    horizontal=True
)
budget_input_value = 1.0 # Default for fixed sum
if budget_type == "固定总和 (Fixed Sum)":
    budget_input_value = st.sidebar.number_input(
        "总保证金比例 (Budget Sum)", value=1.0, min_value=0.0,
        max_value=float(len(available_traders)) if len(available_traders) > 0 else 1.0,
        step=0.01, help="投资组合的总权重之和。例如，1.0代表完全投资。"
    )
    budget_final = budget_input_value
else: # No Sum Constraint
    budget_final = None
    st.sidebar.info("已选择“无总和限制”。权重总和不受约束。")


# --- Model Definitions ---
model_options_all = {
    "最大夏普 (Max Sharpe)": "max_sharpe",
    "最小 CVaR (Min CVaR)": "min_cvar",
    "最小波动率 (Min Variance)": "min_variance",
    "逆波动率 (Inverse Volatility)": "inverse_vol",
    "等权重 (Equal Weighted)": "equal_weighted",
    "分层等风险贡献 (HERC)": "herc",
    "最大化效用 (Maximize Utility)": "max_utility"
}
model_options_sensitive_to_low_variance = [
    "max_sharpe", "min_cvar", "min_variance", "inverse_vol", "herc", "max_utility" # HERC added
]
models_implying_full_investment = ["inverse_vol", "equal_weighted", "herc"]
# Models that are typically not well-defined or less meaningful without a fixed budget sum
models_requiring_fixed_budget = ["min_cvar", "min_variance", "max_utility"]

# Filter models displayed based on budget type and value
model_options_display = model_options_all.copy()

if budget_type == "固定总和 (Fixed Sum)":
    if budget_final != 1.0:
        # Disable models that imply full investment if budget is fixed but not 1.0
        model_options_display = {
            k: v for k, v in model_options_display.items()
            if v not in models_implying_full_investment
        }
elif budget_type == "无总和限制 (No Sum Constraint)": # budget_final is None
    model_options_display = {
        k: v for k, v in model_options_display.items()
        if v not in models_requiring_fixed_budget
    }
# If budget_type is "固定总和 (Fixed Sum)" AND budget_final == 1.0, no filtering is applied by these rules yet,
# so model_options_display remains model_options_all.copy(), which is correct.

default_model_keys = list(model_options_display.keys())


selected_traders = st.sidebar.multiselect(
    "选择带单员", available_traders,
    default=["NovaVault", "YesItsAGreatStrategy", "Solid Trading"] # Keep user's example default
)
# Update the default selection for chosen_model_names based on the filtered display options
chosen_model_names_default = [key for key in default_model_keys[:min(3, len(default_model_keys))]]

chosen_model_names = st.sidebar.multiselect(
    "选择优化模型", list(model_options_display.keys()), # Use keys from the filtered display options
    default=chosen_model_names_default
)
min_w = st.sidebar.slider("最小权重下限 (min_weight)", 0.0, 1.0, 0.0, step=0.01,
                          help="适用于 MeanRisk 类模型。对于 HERC, Inverse Vol, Equal Weighted 模型，此约束不直接适用。")
max_w_min_value = min_w
max_w = st.sidebar.slider("最大权重上限 (max_weight)", float(max_w_min_value), 1.0, 0.5, step=0.01, # Keep user's default of 0.5
                          help="适用于 MeanRisk 类模型。对于 HERC, Inverse Vol, Equal Weighted 模型，此约束不直接适用。")

if min_w > max_w:
    st.sidebar.error("最小权重不能大于最大权重。")
    st.stop()

if st.sidebar.button("🚀 运行优化", use_container_width=True):
    if not selected_traders: st.error("⚠️ 请至少选择一位带单员。"); st.stop()
    if not chosen_model_names: st.error("⚠️ 请至少选择一个优化模型。"); st.stop()

    data_subset = df_prices[selected_traders]
    if len(data_subset) < 2: st.error(f"数据点不足 ({len(data_subset)}) 以计算收益率。"); st.stop()

    X_returns_original = prices_to_returns(data_subset)
    X_returns_original.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_returns_original.isnull().values.any():
        X_returns_original.fillna(method='ffill', inplace=True)
        X_returns_original.fillna(0, inplace=True)
    if len(X_returns_original) < 2: st.error(f"计算收益率后数据点不足 ({len(X_returns_original)})。"); st.stop()
    
    X_train_original, X_test_original = train_test_split(X_returns_original, test_size=0.33, shuffle=False)
    if X_train_original.empty or len(X_train_original) < max(2, len(X_train_original.columns)):
        st.error(f"原始训练集数据过少。"); st.stop()

    variances_on_original_train = X_train_original.var()
    low_variance_threshold = 1e-10
    assets_with_low_variance = variances_on_original_train[variances_on_original_train <= low_variance_threshold].index.tolist()
    assets_for_optimization = variances_on_original_train[variances_on_original_train > low_variance_threshold].index.tolist()
    X_train_for_optimization = X_train_original[assets_for_optimization]
    X_test_for_optimization = X_test_original[assets_for_optimization]

    if assets_with_low_variance:
        st.warning(f"🔔 以下带单员数据在训练期内方差过小: {', '.join(assets_with_low_variance)}。")
        # Check if any chosen sensitive models would be affected by lack of optimizable assets
        chosen_sensitive_model_types = [
            model_options_all[m] for m in chosen_model_names 
            if m in model_options_all and model_options_all[m] in model_options_sensitive_to_low_variance
        ]
        if not assets_for_optimization and chosen_sensitive_model_types:
            st.error("所有选定带单员数据方差均过小，无法运行依赖方差的优化模型。")
            st.stop()
    
    sk_models_to_run = {}
    models_skipped_info = []

    for model_key_name in chosen_model_names:
        model_type = model_options_all.get(model_key_name)
        if not model_type: continue # Should not happen if chosen_model_names comes from model_options_display

        mean_risk_params = {"budget": budget_final, "min_weights": min_w, "max_weights": max_w, "portfolio_params": dict(name=model_key_name)}

        if model_type in model_options_sensitive_to_low_variance and not assets_for_optimization:
            models_skipped_info.append(f"{model_key_name} (无高方差带单员数据)")
            continue
        
        # This check is now implicitly handled by model_options_display, but good for robustness if chosen_model_names was manipulated
        if budget_type == "固定总和 (Fixed Sum)" and budget_final != 1.0 and model_type in models_implying_full_investment:
            models_skipped_info.append(f"{model_key_name} (预算非1.0时，此类模型通常不适用)")
            continue
        if budget_type == "无总和限制 (No Sum Constraint)" and model_type in models_requiring_fixed_budget:
            models_skipped_info.append(f"{model_key_name} (无总和限制时，此类模型通常不适用)")
            continue

        try:
            if model_type == "max_sharpe":
                sk_models_to_run[model_key_name] = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.STANDARD_DEVIATION, **mean_risk_params)
            elif model_type == "min_cvar":
                sk_models_to_run[model_key_name] = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.CVAR, **mean_risk_params)
            elif model_type == "min_variance":
                sk_models_to_run[model_key_name] = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.VARIANCE, **mean_risk_params)
            elif model_type == "max_utility": # Using user's provided call signature
                sk_models_to_run[model_key_name] = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY, risk_measure=RiskMeasure.STANDARD_DEVIATION, **mean_risk_params)
            elif model_type == "inverse_vol":
                 sk_models_to_run[model_key_name] = InverseVolatility(portfolio_params=dict(name=model_key_name))
            elif model_type == "equal_weighted":
                sk_models_to_run[model_key_name] = EqualWeighted(portfolio_params=dict(name=model_key_name))
            elif model_type == "herc":
                sk_models_to_run[model_key_name] = HierarchicalEqualRiskContribution(portfolio_params=dict(name=model_key_name))
            
            if assets_with_low_variance and model_type in model_options_sensitive_to_low_variance:
                 st.info(f"模型 '{model_key_name}' 将仅在方差正常的带单员数据上运行。")

        except Exception as e:
            st.error(f"构建模型 '{model_key_name}' 时出错: {e}")

    if models_skipped_info:
        st.warning(f"以下模型因特定条件未满足而被跳过: {'; '.join(models_skipped_info)}")
    if not sk_models_to_run:
        st.error("没有可运行的优化模型。请检查配置。")
        st.stop()

    fitted_portfolio_objects = {}
    fitted_weights_dict = {}   
    portfolio_returns_dict = {} 
    
    for model_name, model_instance in sk_models_to_run.items():
        model_type = model_options_all[model_name]
        st.write(f"正在优化模型: {model_name}...")

        current_X_train = X_train_for_optimization # Default for sensitive models
        current_X_test = X_test_for_optimization   # Default for sensitive models

        # Models that can/should use all original assets (even with low variance ones)
        if model_type == "equal_weighted" or model_type == "herc": # HERC can also work on original data for clustering
            current_X_train = X_train_original
            current_X_test = X_test_original
            if assets_with_low_variance and model_type == "equal_weighted": # Specific message for EW
                st.info(f"等权重模型 '{model_name}' 将在所有选定带单员数据上计算权重，包括低方差带单员数据。")
            elif assets_with_low_variance and model_type == "herc":
                 st.info(f"HERC模型 '{model_name}' 将在所有选定带单员数据上运行，包括低方差带单员数据。")


        if current_X_train.empty or len(current_X_train.columns) == 0:
            st.warning(f"模型 {model_name} 没有可用的带单员数据进行训练。跳过。")
            fitted_weights_dict[model_name] = pd.Series(dtype=float)
            portfolio_returns_dict[model_name] = np.array([])
            continue
        
        current_asset_names_for_fitting = current_X_train.columns.tolist()

        try:
            model_instance.fit(current_X_train)
            if model_instance.weights_ is None:
                weights_series = pd.Series(np.zeros(len(current_asset_names_for_fitting)), index=current_asset_names_for_fitting)
            else:
                weights_series = pd.Series(model_instance.weights_, index=current_asset_names_for_fitting)
            fitted_weights_dict[model_name] = weights_series

            if not current_X_test.empty and len(current_X_test.columns) > 0 :
                # Ensure predict is called on the same set of assets the model was trained on,
                # or the appropriate corresponding test set.
                predict_X_test = current_X_test
                if model_type == "equal_weighted" or model_type == "herc": # These were fit on X_train_original
                    predict_X_test = X_test_original
                
                if set(predict_X_test.columns) == set(current_asset_names_for_fitting):
                    predicted_portfolio = model_instance.predict(predict_X_test)
                    fitted_portfolio_objects[model_name] = predicted_portfolio
                    portfolio_returns_dict[model_name] = np.asarray(predicted_portfolio.returns)
                elif not predict_X_test.empty: # Fallback if columns don't match but test set is not empty (e.g. due to filtering)
                    st.warning(f"模型 {model_name} 的预测集资产与训练集资产不完全匹配，尝试在可用资产上预测。")
                    # Attempt to predict on the intersection or the filtered test set if applicable
                    predict_on_test_subset = predict_X_test[current_asset_names_for_fitting] if all(c in predict_X_test.columns for c in current_asset_names_for_fitting) else pd.DataFrame()
                    if not predict_on_test_subset.empty:
                        predicted_portfolio = model_instance.predict(predict_on_test_subset)
                        fitted_portfolio_objects[model_name] = predicted_portfolio
                        portfolio_returns_dict[model_name] = np.asarray(predicted_portfolio.returns)
                    else:
                        portfolio_returns_dict[model_name] = np.array([])
                else:
                    portfolio_returns_dict[model_name] = np.array([])
            else:
                portfolio_returns_dict[model_name] = np.array([])

        except OptimizationError as oe:
            st.error(f"优化模型 {model_name} 时出错: {oe}.")
            fitted_weights_dict[model_name] = pd.Series(np.zeros(len(current_asset_names_for_fitting)), index=current_asset_names_for_fitting)
            portfolio_returns_dict[model_name] = np.array([])
        except Exception as e:
            st.error(f"处理模型 {model_name} 时出错: {e}")
            fitted_weights_dict[model_name] = pd.Series(np.zeros(len(current_asset_names_for_fitting)), index=current_asset_names_for_fitting)
            portfolio_returns_dict[model_name] = np.array([])

    st.header("📊 优化结果")
    st.subheader("投资组合权重")
    if fitted_weights_dict:
        # Initialize DataFrame with ALL initially selected traders as columns
        # And with actual models that were run as index
        final_weights_df = pd.DataFrame(index=list(sk_models_to_run.keys()), columns=selected_traders).astype(float)
        for model_name, weights_series in fitted_weights_dict.items():
            if model_name not in final_weights_df.index: continue # Should not happen
            if not weights_series.empty:
                for asset_name, weight_value in weights_series.items():
                    if asset_name in final_weights_df.columns:
                        final_weights_df.loc[model_name, asset_name] = weight_value
        final_weights_df.fillna(0.0, inplace=True)
        formatted_df = final_weights_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        st.dataframe(formatted_df, use_container_width=True)
        if assets_with_low_variance:
            st.caption(f"注意: {', '.join(assets_with_low_variance)} 因方差过小，未被多数优化模型考虑（等权重和HERC模型除外，若运行）。")
    else: st.info("未能计算任何模型的权重。")

    if fitted_portfolio_objects and not X_test_original.empty:
        valid_portfolio_instances = [p for p in fitted_portfolio_objects.values() if p is not None and hasattr(p, 'name')]
        if valid_portfolio_instances:
            try:
                population = Population(valid_portfolio_instances)
                summary_df = population.summary(formatted=True)
                metrics_to_check = ['Annualized Mean', 'Max Drawdown', 'Sharpe Ratio']
                actual_metrics_in_summary = summary_df.index.tolist()
                final_metrics_to_display = []
                for m_check in metrics_to_check:
                    if m_check in actual_metrics_in_summary:
                        final_metrics_to_display.append(m_check)
                    elif m_check == "MAX Drawdown" and "Max Drawdown" in actual_metrics_in_summary: # skfolio uses "Max Drawdown"
                        final_metrics_to_display.append("Max Drawdown")
                
                if final_metrics_to_display:
                    perf_df = summary_df.loc[final_metrics_to_display]
                else:
                    st.warning(f"指定的性能指标 ({', '.join(metrics_to_check)}) 未全部找到。显示所有可用指标。")
                    perf_df = summary_df
                
                st.subheader("模型表现指标")
                st.dataframe(perf_df, use_container_width=True)
            except KeyError as ke:
                st.error(f"提取性能指标时发生KeyError: {ke}。可用指标: {', '.join(summary_df.index.tolist())}")
                if 'summary_df' in locals(): st.dataframe(summary_df, use_container_width=True)
            except Exception as e: st.error(f"计算模型表现指标时出错: {e}")
        else: st.info("没有有效的模型预测结果可用于计算表现指标。")
    elif X_test_original.empty: st.info("测试集为空，因此未计算模型在测试集上的表现指标。")

    if portfolio_returns_dict and not X_test_original.empty:
        st.subheader("累计收益对比 (基于测试集)")
        fig, ax = plt.subplots(figsize=(12, 7))
        non_empty_returns_plotted = False
        for model_name, port_ret_array in portfolio_returns_dict.items():
            model_type = model_options_all.get(model_name)
            # Determine the index for plotting based on which X_test was used for prediction
            plot_index = X_test_for_optimization.index
            if model_type == "equal_weighted": # EW predicts on X_test_original
                plot_index = X_test_original.index
            # For other models, if prediction happened, it was on their respective current_X_test
            # which aligns with X_test_for_optimization for sensitive models.

            if port_ret_array.size > 0 :
                if port_ret_array.ndim > 1: port_ret_array = port_ret_array.squeeze()
                # Ensure length of returns matches the intended plot index
                if len(port_ret_array) == len(plot_index):
                    cumulative_returns = (1 + pd.Series(port_ret_array, index=plot_index)).cumprod() - 1
                    ax.plot(cumulative_returns.index, cumulative_returns.values, label=model_name, linewidth=2)
                    non_empty_returns_plotted = True
                # else: st.warning(f"长度不匹配 {model_name}") # Debug

        for asset_col in X_test_original.columns:
            if not X_test_original[asset_col].empty:
                asset_cumulative_returns = (1 + X_test_original[asset_col]).cumprod() - 1
                ax.plot(asset_cumulative_returns.index, asset_cumulative_returns.values, linestyle='--', alpha=0.7, label=f"{asset_col} (基准)")
        
        if non_empty_returns_plotted or not X_test_original.columns.empty:
            ax.set_title("模型投资组合 vs. 单带单员数据基准 - 累计回报率 (测试集)", fontsize=15)
            ax.set_xlabel("日期"); ax.set_ylabel("累计回报率")
            ax.legend(loc='best', fontsize='small', ncol=max(1, (len(portfolio_returns_dict) + len(X_test_original.columns)) // 6))
            ax.grid(True, linestyle=':', alpha=0.6); plt.xticks(rotation=30, ha='right'); plt.tight_layout(); st.pyplot(fig)
        else: st.info("没有可用于绘制累计收益图的数据。")
    elif X_test_original.empty: st.info("测试集为空，因此未绘制累计收益对比图。")

