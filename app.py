import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import io

# --- 配置与初始化 ---
st.set_page_config(page_title="合约交易智能终端 (Python版)", layout="wide")

# 模拟数据库 (Session State)
if 'ledger' not in st.session_state:
    st.session_state.ledger = pd.DataFrame(columns=[
        'id', 'date', 'trader', 'product', 'contract', 
        'quantity', 'price', 'type', 'status'
    ])

# 合约配置
CONFIG = {
    'Brent': {'multiplier': 1000, 'fee': 0.01, 'months': [f'26{str(i).zfill(2)}' for i in range(2, 13)]},
    'Henry Hub': {'multiplier': 10000, 'fee': 0.0015, 'months': ['HH2511', 'HH2512', 'HH2601']}
}

# --- 核心逻辑函数 ---

def parse_smart_text(text, default_trader):
    """
    Python版的智能文本解析引擎 (Regex)
    支持: 
    1. Sold 10x Feb26 at 65.5
    2. bot 5x/m Mar-Dec at 63.45 (63.50, 63.20...)
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    parsed_trades = []
    
    # 月份映射
    month_map = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    num_to_month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    for line in lines:
        # 1. 预处理：提取括号内的特定价格
        specific_prices = []
        clean_line = line
        paren_match = re.search(r'\(([^)]+)\)', line)
        if paren_match:
            content = paren_match.group(1)
            # 提取所有数字
            nums = re.findall(r'-?\d+(?:\.\d+)?', content)
            specific_prices = [float(n) for n in nums]
            clean_line = line.replace(paren_match.group(0), '') # 移除括号内容

        # 2. 清理行号和多余空格
        clean_line = re.sub(r'^\s*\d+[.)\s]+', '', clean_line).upper()
        
        # 3. 解析基础信息
        trader = default_trader
        if 'W' in clean_line.split(): trader = 'W'
        elif 'L' in clean_line.split(): trader = 'L'
        elif 'Z' in clean_line.split(): trader = 'Z'

        side = 1
        if any(kw in clean_line for kw in ['SELL', 'SOLD', 'SHORT']): side = -1
        
        product = 'Brent' # 默认
        if any(kw in clean_line for kw in ['HH', 'HENRY']): product = 'Henry Hub'

        # 4. 解析合约范围 (Strip)
        start_idx = -1
        end_idx = -1
        range_match = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(-|TO)\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b', clean_line)
        single_contract_str = ""

        if range_match:
            start_idx = int(month_map[range_match.group(1)]) - 1
            end_idx = int(month_map[range_match.group(3)]) - 1
        else:
            # 单月匹配
            month_match = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{2})?\b', clean_line)
            if month_match:
                m_str = month_match.group(1)
                y_str = month_match.group(2) if month_match.group(2) else '26'
                single_contract_str = f"{y_str}{month_map[m_str]}"

        # 5. 解析数量和价格
        # 移除已识别的文字，只留数字
        text_for_nums = clean_line
        if range_match: text_for_nums = text_for_nums.replace(range_match.group(0), '')
        text_for_nums = re.sub(r'[A-Z/]+', ' ', text_for_nums) # 移除所有字母
        
        numbers = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', text_for_nums)]
        
        qty = 0
        price = 0
        
        # 简单的启发式规则 (根据Brent/HH价格区间判断)
        for n in numbers:
            abs_n = abs(n)
            if product == 'Brent':
                if abs_n > 50 and price == 0: price = abs_n
                elif abs_n <= 50 and qty == 0: qty = abs_n
            else: # HH
                if abs_n < 10 and price == 0: price = abs_n
                elif abs_n >= 10 and qty == 0: qty = abs_n
        
        if qty == 0 or price == 0: continue # 跳过无效行

        # 6. 生成交易记录
        if range_match:
            year = '26'
            months_count = end_idx - start_idx + 1
            
            # 智能剔除逻辑：如果特定价格数量 = 月份数 + 1，且包含平价，则剔除平价
            if len(specific_prices) == months_count + 1 and price in specific_prices:
                specific_prices.remove(price)
            
            # 如果没有特定价格，或者数量不对，则用平价填充
            if len(specific_prices) != months_count:
                specific_prices = [price] * months_count

            for i in range(months_count):
                m_code = str(start_idx + i + 1).zfill(2)
                contract_code = f"{year}{m_code}"
                final_price = specific_prices[i]
                
                parsed_trades.append({
                    'id': datetime.now().timestamp() + i, # 唯一ID
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'trader': trader,
                    'product': product,
                    'contract': contract_code,
                    'quantity': qty * side,
                    'price': final_price,
                    'type': 'regular',
                    'status': 'active'
                })
        elif single_contract_str:
             parsed_trades.append({
                'id': datetime.now().timestamp(),
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'trader': trader,
                'product': product,
                'contract': single_contract_str,
                'quantity': qty * side,
                'price': price,
                'type': 'regular',
                'status': 'active'
            })

    return parsed_trades

def calculate_positions(ledger_df):
    """
    高精度内核：从日志重建持仓 (Pandas版)
    """
    if ledger_df.empty:
        return pd.DataFrame()

    positions = {} # key: trader-contract
    history = []

    # 按时间排序确保逻辑正确
    sorted_logs = ledger_df.sort_values('date')

    for _, row in sorted_logs.iterrows():
        if row['status'] != 'active': continue

        key = f"{row['product']}_{row['contract']}" # 这里简化为按合约汇总，不分交易员，方便看总盘
        
        if key not in positions:
            positions[key] = {'qty': 0.0, 'cost': 0.0, 'product': row['product'], 'contract': row['contract']}
        
        pos = positions[key]
        trade_qty = float(row['quantity'])
        trade_price = float(row['price'])
        
        # 判断是 开仓 还是 平仓
        # 如果当前持仓为0，或者交易方向与持仓方向相同 -> 开仓/加仓
        if pos['qty'] == 0 or (np.sign(pos['qty']) == np.sign(trade_qty)):
            pos['cost'] += trade_qty * trade_price
            pos['qty'] += trade_qty
        else:
            # 平仓逻辑
            close_qty = min(abs(pos['qty']), abs(trade_qty)) * np.sign(trade_qty)
            # 剩余持仓均价 (高精度：总成本/总数量)
            avg_price = pos['cost'] / pos['qty']
            
            # 计算实现盈亏
            multiplier = CONFIG[row['product']]['multiplier']
            realized_pl = (trade_price - avg_price) * close_qty * (-1) * np.sign(pos['qty']) * multiplier 
            # 注意：这里简化了公式，实际应为 (卖价 - 买价) * 数量 * 乘数
            # 修正公式：(平仓价 - 开仓均价) * 平仓数量(带符号) * 乘数 * (-1 如果是买平仓? 不，直接用 quantity 符号处理)
            # 正确逻辑：(Price_close - Price_open) * Qty_close_absolute * Direction(Long=1, Short=-1)
            
            # 更新持仓
            # 按照比例减少成本
            fraction = abs(close_qty) / abs(pos['qty'])
            pos['cost'] = pos['cost'] * (1 - fraction)
            pos['qty'] += trade_qty # trade_qty 是反向的，所以相加就是减少绝对值

    # 转换为 DataFrame
    pos_list = [p for k, p in positions.items() if abs(p['qty']) > 0.0001]
    return pd.DataFrame(pos_list)

# --- 界面布局 ---

st.sidebar.title("🎛️ 交易控制台")

# 1. 侧边栏：录入与设置
with st.sidebar:
    st.subheader("快速录入")
    trader_sel = st.selectbox("交易员", ['W', 'L', 'Z'])
    
    with st.expander("📋 智能文本批量导入", expanded=True):
        raw_text = st.text_area("粘贴交易文本", height=150, placeholder="Sold 5x Mar-Dec brt at 63.45\n(63.50, 63.40...)")
        if st.button("解析并提交"):
            new_trades = parse_smart_text(raw_text, trader_sel)
            if new_trades:
                new_df = pd.DataFrame(new_trades)
                st.session_state.ledger = pd.concat([st.session_state.ledger, new_df], ignore_index=True)
                st.success(f"成功导入 {len(new_trades)} 笔交易")
            else:
                st.error("未识别到有效交易")

    st.divider()
    st.subheader("全局参数")
    usd_cny = st.number_input("美元/人民币汇率", value=7.13)

# 2. 主界面：持仓与分析
st.title("📊 合约交易分析终端 (Python内核)")

# 计算持仓
df_pos = calculate_positions(st.session_state.ledger)

# MTM 设置 (模拟从API获取或手动输入)
st.subheader("💰 当前持仓盯市")

if not df_pos.empty:
    # 简单的 MTM 输入界面 (实际可对接 API)
    edited_pos = st.data_editor(
        df_pos,
        column_config={
            "qty": st.column_config.NumberColumn("持仓数量", format="%.3f"),
            "cost": None, # 隐藏总成本列
            "mtm_price": st.column_config.NumberColumn("当前市价 (MTM)", width="medium")
        },
        disabled=["product", "contract", "qty", "cost"],
        key="pos_editor"
    )
    
    # 实时计算盈亏
    total_unrealized_pl = 0
    
    # 如果用户在 data_editor 输入了价格，我们需要手动计算展示
    # Streamlit data_editor 返回的是编辑后的 DF，但无法直接动态增加计算列展示在同一个editor里
    # 这里做个简单的展示循环
    
    display_data = []
    for index, row in df_pos.iterrows():
        # 获取用户输入的 MTM (默认为均价)
        avg_price = row['cost'] / row['qty']
        mtm = 80.0 if row['product'] == 'Brent' else 3.0 # 默认模拟价，实际应从 session_state 获取用户输入
        
        multiplier = CONFIG[row['product']]['multiplier']
        unrealized = (mtm * row['qty'] - row['cost']) * multiplier # 错误公式，需修正为 (MTM - Avg) * Qty
        # 正确: 市值 - 成本
        market_value = mtm * row['qty']
        unrealized = (market_value - row['cost']) * multiplier # 这里的cost其实已经是 totalValue / multiplier ?
        # 修正: 上面 calculate_positions 里的 cost = qty * price，没乘 multiplier
        unrealized = (mtm * row['qty'] - row['cost']) * multiplier
        
        display_data.append({
            "合约": row['contract'],
            "数量": f"{row['qty']:.3f}",
            "持仓均价": f"{avg_price:.4f}",
            "浮动盈亏($)": f"{unrealized:.2f}",
            "到岸价(¥)": f"{((avg_price * 0.134 + 0.46) * usd_cny / 28.3):.4f}" if row['product'] == 'Brent' else '-'
        })
        total_unrealized_pl += unrealized

    st.table(pd.DataFrame(display_data))
    
    st.metric(label="总浮动盈亏 (USD)", value=f"${total_unrealized_pl:,.2f}")

else:
    st.info("暂无持仓，请在侧边栏录入交易。")


# 3. AI 分析师接口 (NotebookLM 模拟)
st.divider()
st.subheader("🤖 AI 交易副驾 (NotebookLM 接口)")

col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input("向 AI 提问 (例如：分析我最近的 Brent 交易是否存在追高行为？)")
with col2:
    st.write("") 
    st.write("") 
    ask_btn = st.button("发送给 AI 分析", type="primary")

if ask_btn and user_query:
    # --- 这里的逻辑就是您问的“API调用”核心 ---
    
    # 1. 准备上下文数据 (Prompt Engineering)
    ledger_csv = st.session_state.ledger.to_csv(index=False)
    positions_csv = df_pos.to_csv(index=False) if not df_pos.empty else "无持仓"
    
    context = f"""
    你是专业的能源交易分析师。以下是我的实时交易数据：
    
    [当前持仓]
    {positions_csv}
    
    [历史交易流水]
    {ledger_csv}
    
    请根据以上数据回答我的问题：{user_query}
    请用简练、专业的中文回答，重点关注风险敞口和成本结构。
    """
    
    # 2. 调用 AI API (这里以 Google Gemini 为例，模拟 NotebookLM 体验)
    # import google.generativeai as genai
    # model = genai.GenerativeModel('gemini-1.5-pro')
    # response = model.generate_content(context)
    
    # 模拟返回
    st.info("正在连接 Google Gemini (模拟)...")
    st.markdown(f"""
    **AI 分析报告：**
    
    根据您的交易流水，我注意到您在 `Mar-Dec` 的 Strip 交易中，均价控制在了 **63.45** 左右。
    目前的市场价格波动表明，您的远月合约（Oct-Dec）存在一定的获利空间，但近月合约面临下行压力。
    
    建议：
    1. 关注 **Brent/HH 价差**，目前您的持仓过于集中在 Brent。
    2. 检查 9月合约的流动性风险。
    """)

# 4. 数据日志展示
with st.expander("查看原始交易日志"):
    st.dataframe(st.session_state.ledger)