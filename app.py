import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import re
import requests
from datetime import datetime, timedelta, timezone

# Unified UTC reference compatible across Python versions
UTC = getattr(datetime, 'UTC', timezone.utc)


# ----------------------- Configuration Constants -----------------------

# List of traders available in the system.  The default trader selected
# in the UI will be used when parsing free‑form trade strings where no
# explicit trader is mentioned.
TRADERS = ['D', 'L', 'Z']

# Contract codes by product.  Brent contracts are labelled with a
# four‑digit code, while Henry Hub contracts are prefaced with ``HH``.
# These lists are used to populate drop‑downs and provide sensible
# defaults when importing market data.
CONTRACTS = {
    'Brent': ['2602', '2603', '2604', '2605', '2606', '2607', '2608', '2609', '2610', '2611', '2612'],
    'Henry Hub': ['HH2511', 'HH2512', 'HH2601'],
}

# Contract multipliers translate a one‑lot position into the number of
# underlying units.  Brent lots correspond to 1 000 barrels and
# Henry Hub lots correspond to 10 000 MMBtu.
CONTRACT_MULTIPLIERS = {
    'Brent': 1000,
    'Henry Hub': 10000,
}

# Colour palette used in the charts.  The colours are defined using
# RGBA strings so that transparency can be controlled when plotting
# stacked or overlapping objects.
COLOURS = {
    'Brent': 'rgba(59, 130, 246, 0.7)',      # blue
    'Henry Hub': 'rgba(16, 185, 129, 0.7)',   # green
}

# Mapping from month abbreviations to two‑digit numbers.  This is
# identical to the JavaScript version and is used when parsing month
# based contract descriptions such as “Feb 26” or “26‑Feb”.
MONTH_MAP = {
    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
}

# The reverse mapping is useful for generating human readable labels.
NUM_TO_MONTH = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# DeepSeek API configuration
DEFAULT_DEEPSEEK_MODEL = 'deepseek-chat'

# DeepSeek API configuration
DEFAULT_DEEPSEEK_MODEL = 'deepseek-chat'

# DeepSeek API configuration
DEFAULT_DEEPSEEK_MODEL = 'deepseek-chat'

# DeepSeek API configuration
DEFAULT_DEEPSEEK_MODEL = 'deepseek-chat'


def init_session_state() -> None:
    """Initialise the session state keys if they are missing.

    Streamlit reruns your script from top to bottom whenever a widget
    changes, so it’s important to persist data between runs.  We do
    this by storing data structures in ``st.session_state``.  On the
    first run the session state dictionary is empty, so we create the
    keys and assign sensible defaults.
    """
    state_defaults = {
        'positions': [],       # list of dicts describing open positions
        'history': [],         # list of dicts describing realised P/L events
        'transaction_log': [], # list of dicts describing every trade (active or reversed)
        'market_prices': {},   # dict mapping contract to MTM price
        'settings': {
            'fees': {
                'brent_per_bbl': 0.00,
                'hh_per_mmbtu': 0.0000
            },
            'exchange_rate_rmb': 7.13,
            'initial_realised_pl': 0.00
        },
        'parsed_trades_buffer': [],  # temporary buffer for batch import preview
        'last_selected_trader': TRADERS[0],  # remember last trader for parsing
        'show_batch_import': False,  # whether the batch import modal is visible
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_price(price: float, product: str) -> str:
    """Format a price for display based on the product’s precision.

    Brent prices are normally quoted to 2 decimal places whereas Henry
    Hub prices are quoted to 4 decimal places.  If the price is
    ``None`` or not a number, return ``'--'`` to indicate that no
    current price is available.

    Args:
        price: The numeric price to format.
        product: Either ``'Brent'`` or ``'Henry Hub'``.

    Returns:
        A string representation of the price with appropriate precision.
    """
    if price is None or isinstance(price, str) and not price:
        return '--'
    try:
        price = float(price)
    except (ValueError, TypeError):
        return '--'
    precision = 4 if product == 'Henry Hub' else 2
    return f"{price:.{precision}f}"


def rebuild_state_from_logs() -> None:
    """Recompute positions and history from the active transaction log.

    Whenever a trade is entered or reversed the positions and history
    derived from the transaction log must be recomputed.  This function
    reads ``st.session_state.transaction_log``, filters out reversed
    trades and then iterates over them in chronological order to build
    up the positions and realised P/L history.  It writes the results
    back into ``st.session_state.positions`` and
    ``st.session_state.history``.
    """
    logs = [log for log in st.session_state['transaction_log'] if log.get('status', 'active') == 'active']
    logs.sort(key=lambda l: l['date'])
    positions = {}
    history = []
    settings = st.session_state['settings']

    for log in logs:
        trader = log['trader']
        product = log['product']
        contract = log['contract']
        qty = log['quantity']
        price = log['price']
        trade_type = log.get('type', 'regular')
        key = f"{trader}-{contract}"
        pos = positions.get(key, {'trader': trader, 'product': product, 'contract': contract, 'quantity': 0.0, 'total_value': 0.0})
        # Average price of the existing position
        avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0

        # Detect closing trades (sign change) – quantity and price both matter
        if abs(pos['quantity']) > 1e-12 and np.sign(pos['quantity']) != np.sign(qty):
            close_qty = min(abs(pos['quantity']), abs(qty))
            direction = np.sign(pos['quantity'])
            # Realised P/L only for regular trades
            if trade_type == 'regular':
                gross_pl = (price - avg_price) * close_qty * direction * CONTRACT_MULTIPLIERS[product]
                fee_per_unit = settings['fees']['brent_per_bbl'] if product == 'Brent' else settings['fees']['hh_per_mmbtu']
                commission_cost = close_qty * CONTRACT_MULTIPLIERS[product] * 2 * fee_per_unit
                history.append({
                    'date': log['date'],
                    'trader': trader,
                    'product': product,
                    'contract': contract,
                    'closed_quantity': close_qty * -direction,  # negative for selling to close long
                    'open_price': avg_price,
                    'close_price': price,
                    'realised_pl': gross_pl - commission_cost
                })
                # Reduce the position by the closed quantity
                pos['total_value'] = avg_price * (pos['quantity'] + qty)
                pos['quantity'] += qty
            else:
                # Adjustment trades modify cost basis without realising P/L
                adjustment_pl = (price - avg_price) * close_qty * direction
                remaining_qty = pos['quantity'] + qty
                pos['total_value'] = avg_price * remaining_qty - adjustment_pl
                pos['quantity'] = remaining_qty
        else:
            # Opening trade; simply accumulate
            pos['total_value'] += qty * price
            pos['quantity'] += qty
        positions[key] = pos

    # Convert dict to list, filter out flat positions
    st.session_state['positions'] = [p for p in positions.values() if abs(p['quantity']) > 1e-9]
    st.session_state['history'] = history


def add_transaction(trader: str, product: str, contract: str, quantity: float, price: float, trade_type: str = 'regular') -> None:
    """Add a single transaction to the log and recompute state.

    Args:
        trader: One of the predefined trader codes (e.g. ``'W'``).
        product: ``'Brent'`` or ``'Henry Hub'``.
        contract: The four or six digit contract code.
        quantity: Positive for buy, negative for sell.  Quantities are
            always measured in lots, not in barrels/MMBtu.
        price: Price per unit (USD per barrel or per MMBtu).
        trade_type: Either ``'regular'`` or ``'adjustment'`` to
            distinguish between normal trades and cost adjustments.
    """
    st.session_state['transaction_log'].append({
        'id': float(datetime.now(UTC).timestamp()) + np.random.random(),
        'date': datetime.now(UTC).isoformat(),
        'trader': trader,
        'product': product,
        'contract': contract,
        'quantity': quantity,
        'price': price,
        'status': 'active',
        'type': trade_type,
    })
    rebuild_state_from_logs()


def reverse_transaction(log_id: float) -> None:
    """Mark a transaction as reversed and recompute positions/history."""
    for log in st.session_state['transaction_log']:
        if log['id'] == log_id:
            log['status'] = 'reversed'
            break
    rebuild_state_from_logs()


def parse_trade_line(line: str, default_trader: str) -> list:
    """Parse a single free‑form trade description into one or more trades.

    This function mirrors the logic of the JavaScript ``parseLine``
    function.  It attempts to recognise traders, product keywords,
    contract codes, quantity indicators, price points, ranges and lists
    of prices.  The output is a list of dictionaries, one per trade.
    Each dictionary contains the keys: ``trader``, ``product``,
    ``contract``, ``side`` (1 for buy, –1 for sell), ``qty`` (lots),
    ``price`` (USD per unit) and ``final_qty`` (qty × side).  The
    ``is_valid`` key flags whether the parser believes the line is
    sufficiently well formed to be executed.

    Args:
        line: The raw input line from the user.
        default_trader: The trader code selected in the form (used if
            the line itself does not mention a trader).

    Returns:
        A list of parsed trade dictionaries.  Invalid parses will have
        ``is_valid`` set to ``False`` and missing fields filled with
        ``None``.
    """
    # Prepare output list
    results = []
    try:
        # Extract parenthetical price lists e.g. "(61.43 61.22 ...)" and remove from line
        specific_prices = []
        parens_match = re.search(r'\(([^)]+)\)', line)
        clean_line = line
        if parens_match:
            content = parens_match.group(1)
            at_split = re.split(r'(?i)at\s+', content)
            numbers_part = at_split[1] if len(at_split) > 1 else content
            extracted_nums = re.findall(r'-?\d+(?:\.\d+)?', numbers_part)
            if extracted_nums:
                specific_prices = [float(n) for n in extracted_nums]
            clean_line = line[:parens_match.start()] + line[parens_match.end():]

        # Remove leading enumerations like "1. " or "57)"
        clean_line = re.sub(r'^\s*\d+[.)\s]+', '', clean_line)
        upper_line = clean_line.upper()

        # Determine trader
        trader = default_trader
        if re.search(r'\bW\b', upper_line):
            trader = 'W'
        elif re.search(r'\bL\b', upper_line):
            trader = 'L'
        elif re.search(r'\bZ\b', upper_line):
            trader = 'Z'

        # Determine side (buy/sell)
        side = 1
        if re.search(r'SELL|SOLD|SHORT|卖|平', upper_line):
            side = -1
        elif re.search(r'BOT|BOUGHT|BUY|LONG|买|建', upper_line):
            side = 1

        # Determine product
        product = ''
        if re.search(r'HH|HENRY|HUB', upper_line):
            product = 'Henry Hub'
        elif re.search(r'BRT|BRENT', upper_line) or re.search(r'\b(25|26)\d{2}\b', upper_line):
            product = 'Brent'

        # Detect month range e.g. "MAR-DEC" or "APR TO JUN"
        range_match = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(-|TO)\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b', upper_line)
        start_month_idx = -1
        end_month_idx = -1
        single_contract = ''
        matched_contract_string = ''

        if range_match:
            start_month = range_match.group(1)
            end_month = range_match.group(3)
            start_month_idx = int(MONTH_MAP[start_month]) - 1
            end_month_idx = int(MONTH_MAP[end_month]) - 1
            matched_contract_string = range_match.group(0)
            if not product:
                product = 'Brent'
        else:
            # Look for explicit contract codes
            hh_code_match = re.search(r'HH\d{4}', upper_line)
            brent_code_match = re.search(r'\b(25|26)\d{2}\b', upper_line)
            month_year_match1 = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{2})?\b', upper_line)
            month_year_match2 = re.search(r'\b(\d{2})-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b', upper_line)
            if hh_code_match:
                product = 'Henry Hub'
                single_contract = hh_code_match.group(0)
                matched_contract_string = hh_code_match.group(0)
            elif brent_code_match:
                product = 'Brent'
                single_contract = brent_code_match.group(0)
                matched_contract_string = brent_code_match.group(0)
            elif month_year_match2:
                # Format "26-FEB"
                if not product:
                    product = 'Brent'
                y_str = month_year_match2.group(1)
                m_str = month_year_match2.group(2)
                single_contract = y_str + MONTH_MAP[m_str]
                matched_contract_string = month_year_match2.group(0)
            elif month_year_match1:
                # Format "FEB" or "FEB 26"
                if not product:
                    product = 'Brent'
                m_str = month_year_match1.group(1)
                y_str = month_year_match1.group(2) or '26'
                single_contract = y_str + MONTH_MAP[m_str]
                matched_contract_string = month_year_match1.group(0)

        # Build a string for extracting numeric quantities/prices
        text_for_nums = upper_line
        if matched_contract_string:
            text_for_nums = text_for_nums.replace(matched_contract_string, '')
        # Remove common tokens
        text_for_nums = re.sub(r'BRT|BRENT|HH|HENRY|HUB|SOLD|SELL|SHORT|BOT|BOUGHT|BUY|LONG|PM|OTC|AT|KB|LOTS?', '', text_for_nums)
        # Extract all numbers
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text_for_nums)
        qty = 0.0
        price = 0.0

        # Attempt to find quantity specified with X or LOTS/KB
        qty_x_match = re.search(r'(\d+(?:\.\d+)?)\s*X', upper_line)
        qty_lots_match = re.search(r'(\d+(?:\.\d+)?)\s*(LOTS?|KB)', upper_line)
        if qty_x_match:
            qty = float(qty_x_match.group(1))
        elif qty_lots_match:
            qty = float(qty_lots_match.group(1))

        # Attempt to find price explicitly with "AT"
        price_at_match = re.search(r'AT\s*(\d+(?:\.\d+)?)', upper_line)
        if price_at_match:
            price = float(price_at_match.group(1))

        # Now use remaining numbers to infer quantity/price if still missing
        nums = [float(n) for n in numbers]
        remaining = nums.copy()
        # Remove quantity if found in pattern
        if qty > 0.0:
            remaining = [n for n in remaining if abs(n - qty) > 1e-9]
        # Remove price if found via AT
        if price > 0.0:
            remaining = [n for n in remaining if abs(n - price) > 1e-9]
        # Remove numbers that appear in specific price list
        if specific_prices:
            remaining = [n for n in remaining if not any(abs(n - p) < 1e-9 for p in specific_prices)]

        # If we still haven't got qty/price try to infer them
        if qty == 0.0 or price == 0.0:
            if len(remaining) >= 2 and qty == 0.0 and price == 0.0:
                n1, n2 = abs(remaining[0]), abs(remaining[1])
                # Heuristic: Henry Hub price < 20, quantity likely bigger
                if product == 'Henry Hub':
                    if n1 < 20 and n2 >= 10:
                        price, qty = n1, n2
                    else:
                        qty, price = n1, n2
                else:  # Brent
                    if n1 > 50:
                        price, qty = n1, n2
                    else:
                        qty, price = n1, n2
            elif len(remaining) >= 1:
                if qty == 0.0 and price > 0.0:
                    qty = abs(remaining[0])
                elif price == 0.0 and qty > 0.0:
                    price = abs(remaining[0])
                elif qty == 0.0 and price == 0.0:
                    # Only one number – assume it’s quantity
                    qty = abs(remaining[0])

        # If after all heuristics qty or price is still zero and no specific prices, mark invalid
        is_valid = True
        if qty == 0.0 or (price == 0.0 and not specific_prices):
            is_valid = False

        # Process month range
        if range_match and is_valid:
            year = '26'
            total_months = end_month_idx - start_month_idx + 1
            # Try to eliminate flat price from specific price list (if length equals months+1)
            sp = specific_prices.copy()
            if sp:
                if len(sp) == total_months + 1 and price != 0.0:
                    # Remove the element equal to price
                    for idx, p in enumerate(sp):
                        if abs(p - price) < 1e-9:
                            sp.pop(idx)
                            break
                elif len(sp) == total_months + 1:
                    # Remove first element
                    sp = sp[1:]
            for i in range(total_months):
                month_code = str(start_month_idx + i + 1).zfill(2)
                contract = year + month_code
                # Determine final price for each leg
                final_price = price
                if sp:
                    if len(sp) >= total_months:
                        final_price = sp[i]
                    else:
                        final_price = sp[i] if i < len(sp) else price
                final_qty = qty * side
                results.append({
                    'trader': trader,
                    'product': product,
                    'contract': contract,
                    'side': side,
                    'qty': qty,
                    'price': final_price,
                    'final_qty': final_qty,
                    'is_valid': is_valid,
                })
        else:
            # Single contract
            final_qty = qty * side
            final_price = price if price != 0.0 else (specific_prices[0] if specific_prices else price)
            results.append({
                'trader': trader,
                'product': product,
                'contract': single_contract,
                'side': side,
                'qty': qty,
                'price': final_price,
                'final_qty': final_qty,
                'is_valid': is_valid and bool(single_contract)
            })
    except Exception:
        # Fail gracefully
        results.append({
            'trader': None,
            'product': None,
            'contract': None,
            'side': 1,
            'qty': 0.0,
            'price': 0.0,
            'final_qty': 0.0,
            'is_valid': False
        })
    return results


def parse_batch_input(text: str, default_trader: str) -> list:
    """Parse multi‑line input for batch import and return trade list.

    The original application merges lines consisting solely of numbers
    into the preceding line to support strip price lists.  This
    behaviour is mirrored here.  Each parsed trade includes an
    ``is_valid`` flag.

    Args:
        text: Raw multiline string pasted by the user.
        default_trader: Trader code selected in the UI.

    Returns:
        A list of parsed trades (each as a dict) ready for preview.
    """
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    merged_lines = []
    for line in raw_lines:
        stripped = line.strip()
        # Detect if the line appears to be a list of prices: consists of numbers and spaces, no letters
        is_price_list = bool(re.match(r'^\s*(\d+(\.\d+)?(\s+|$))+', stripped)) and not re.search(r'[A-Za-z]', stripped)
        if is_price_list and merged_lines:
            merged_lines[-1] += ' ' + stripped
        else:
            merged_lines.append(stripped)
    all_trades = []
    for ln in merged_lines:
        trades = parse_trade_line(ln, default_trader)
        all_trades.extend(trades)
    return all_trades


def export_json() -> str:
    """Return a JSON string representing the entire application state."""
    state = {
        'positions': st.session_state['positions'],
        'history': st.session_state['history'],
        'transaction_log': st.session_state['transaction_log'],
        'market_prices': st.session_state['market_prices'],
        'settings': st.session_state['settings'],
    }
    return json.dumps(state, indent=2, default=str)


def import_json(json_str: str) -> bool:
    """Import application state from a JSON string.

    The JSON must include ``transaction_log``; all other keys are
    optional but will overwrite the current session state.  Returns
    ``True`` if import succeeded, otherwise ``False``.
    """
    try:
        data = json.loads(json_str)
    except Exception:
        return False
    if 'transaction_log' not in data:
        return False
    st.session_state['transaction_log'] = data.get('transaction_log', [])
    st.session_state['market_prices'] = data.get('market_prices', {})
    st.session_state['settings'] = data.get('settings', st.session_state['settings'])
    # Rebuild positions/history based on imported logs
    rebuild_state_from_logs()
    return True


def import_mtm_json(json_str: str) -> int:
    """Import MTM prices from a JSON string.  Returns number updated."""
    try:
        data = json.loads(json_str)
    except Exception:
        return 0
    market_prices = data.get('market_prices')
    if not isinstance(market_prices, dict):
        return 0
    count = 0
    for contract, price in market_prices.items():
        try:
            st.session_state['market_prices'][contract] = float(price)
            count += 1
        except (ValueError, TypeError):
            continue
    return count


def export_positions_csv() -> str:
    """Generate CSV string for current positions table."""
    rows = []
    for pos in st.session_state['positions']:
        product = pos['product']
        avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
        current_price = st.session_state['market_prices'].get(pos['contract'], avg_price)
        gross_pl = (current_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
        fee_per_unit = st.session_state['settings']['fees']['brent_per_bbl'] if product == 'Brent' else st.session_state['settings']['fees']['hh_per_mmbtu']
        commission = abs(pos['quantity']) * CONTRACT_MULTIPLIERS[product] * fee_per_unit
        floating_pl = gross_pl - commission
        # Landed price (approximate cost converted to RMB) – replicates JS logic
        rmb = st.session_state['settings']['exchange_rate_rmb'] or 7.13
        landed_price = 0.0
        if product == 'Brent':
            landed_price = (avg_price * 0.134 + 0.46) * rmb / 28.3
        elif product == 'Henry Hub':
            landed_price = (avg_price * 1.15 + 4.5) * rmb / 28.3
        rows.append({
            '合约': pos['contract'],
            '数量': f"{pos['quantity']:.3f}",
            '均价': format_price(avg_price, product),
            'MTM价格': format_price(current_price, product),
            '浮动净P/L': f"{floating_pl:.2f}",
            '对应到岸价': f"{landed_price:.4f}" if landed_price > 0 else ''
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8-sig')


def export_history_csv() -> str:
    """Generate CSV string for realised P/L history."""
    rows = []
    initial_pl = st.session_state['settings'].get('initial_realised_pl', 0.0)
    total = initial_pl
    for h in sorted(st.session_state['history'], key=lambda x: x['date']):
        total += h['realised_pl']
        rows.append({
            '日期': h['date'].split('T')[0],
            '交易员': h['trader'],
            '合约': h['contract'],
            '平仓量': f"{h['closed_quantity']:.3f}",
            '开仓价': format_price(h['open_price'], h['product']),
            '平仓价': format_price(h['close_price'], h['product']),
            '实现净P/L': f"{h['realised_pl']:.2f}"
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8-sig')


def export_log_csv() -> str:
    """Generate CSV string for the transaction log."""
    month_map_for_export = {
        '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'may',
        '06': 'jun', '07': 'jul', '08': 'aug', '09': 'sep', '10': 'oct',
        '11': 'nov', '12': 'dec'
    }
    def get_contract_month(contract_code: str) -> str:
        if len(contract_code) == 4 and contract_code.isdigit():
            return month_map_for_export.get(contract_code[2:4], '')
        if contract_code.startswith('HH') and len(contract_code) == 6:
            return month_map_for_export.get(contract_code[4:6], '')
        return ''
    rows = []
    counter = 1
    for log in st.session_state['transaction_log']:
        if log.get('status') != 'active':
            continue
        trade_type_name = '成本调整' if log.get('type') == 'adjustment' else '常规交易'
        rows.append({
            '时间': datetime.fromisoformat(log['date']).strftime('%Y-%m-%d %H:%M:%S'),
            '编号': counter,
            '成交品种': log['product'],
            '交易类型': trade_type_name,
            '合约月份': get_contract_month(log['contract']),
            '成交数量': abs(log['quantity']),
            '成交价格': format_price(log['price'], log['product'])
        })
        counter += 1
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding='utf-8-sig')


def scenario_analysis(delta_brent: float, delta_hh: float) -> tuple:
    """Perform stress test and return delta P/L and new total unrealised P/L.

    Args:
        delta_brent: Price change applied to all Brent positions.
        delta_hh: Price change applied to all Henry Hub positions.

    Returns:
        A tuple of (pl_change, new_total_pl).  Both values are floats.
    """
    current_total_pl = 0.0
    hypothetical_total_pl = 0.0
    for pos in st.session_state['positions']:
        product = pos['product']
        avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
        current_price = st.session_state['market_prices'].get(pos['contract'], avg_price)
        gross_pl = (current_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
        fee_per_unit = st.session_state['settings']['fees']['brent_per_bbl'] if product == 'Brent' else st.session_state['settings']['fees']['hh_per_mmbtu']
        commission_cost = abs(pos['quantity']) * CONTRACT_MULTIPLIERS[product] * fee_per_unit
        current_total_pl += gross_pl - commission_cost
        # Apply price change
        price_change = delta_brent if product == 'Brent' else delta_hh
        hypothetical_price = current_price + price_change
        hypo_gross_pl = (hypothetical_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
        hypothetical_total_pl += hypo_gross_pl - commission_cost
    pl_change = hypothetical_total_pl - current_total_pl
    return pl_change, hypothetical_total_pl


def build_infographics() -> tuple:
    """Generate Altair charts for the position structure and realised P/L."""
    # Position structure pie chart
    pos_df = pd.DataFrame([
        {
            'product': p['product'],
            'value': abs(p['quantity'] * (p['total_value'] / p['quantity'] if abs(p['quantity']) > 1e-12 else 0.0) * CONTRACT_MULTIPLIERS[p['product']]),
        }
        for p in st.session_state['positions']
    ])
    if not pos_df.empty:
        pos_agg = pos_df.groupby('product', as_index=False)['value'].sum()
        pie_chart = alt.Chart(pos_agg).mark_arc(innerRadius=40).encode(
            theta='value:Q',
            color=alt.Color('product:N', scale=alt.Scale(domain=list(COLOURS.keys()), range=[COLOURS[p] for p in COLOURS])),
            tooltip=['product:N', alt.Tooltip('value:Q', format=',.2f')]
        ).properties(height=300)
    else:
        pie_chart = alt.Chart(pd.DataFrame({'placeholder': [0]})).mark_text(text='No positions').properties(height=300)
    # Realised P/L line chart
    initial = st.session_state['settings'].get('initial_realised_pl', 0.0)
    history_sorted = sorted(st.session_state['history'], key=lambda x: x['date'])
    dates = []
    cums = []
    cum_pl = initial
    # include a zero point one day before first trade
    if history_sorted:
        first_date = datetime.fromisoformat(history_sorted[0]['date']).date()
        dates.append((first_date - pd.Timedelta(days=1)).isoformat())
        cums.append(cum_pl)
    for h in history_sorted:
        cum_pl += h['realised_pl']
        dates.append(h['date'][:10])
        cums.append(cum_pl)
    if not dates:
        # no history; just show a flat line at initial
        dates = [datetime.now().date().isoformat()]
        cums = [initial]
    pl_df = pd.DataFrame({'date': dates, 'cumulative_pl': cums})
    pl_chart = alt.Chart(pl_df).mark_line().encode(
        x='date:T',
        y=alt.Y('cumulative_pl:Q', title='Cumulative Realised P/L'),
        tooltip=[alt.Tooltip('date:T'), alt.Tooltip('cumulative_pl:Q', format=',.2f')]
    ).properties(height=300)
    return pie_chart, pl_chart


def load_ticket_data(uploaded_file) -> tuple:
    """Load a ticket file (CSV/XLSX) into a DataFrame.

    Returns a tuple of (DataFrame or None, error message).
    """
    if uploaded_file is None:
        return None, '请先上传水单文件。'
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            return None, '仅支持 CSV 或 XLSX 格式的水单明细。'
        return df, ''
    except Exception as exc:  # pragma: no cover - UI feedback only
        return None, f'读取水单失败: {exc}'


def reconcile_tickets(ticket_df: pd.DataFrame, start_date, end_date) -> dict:
    """Compare uploaded ticket rows against system logs for a date window."""

    def _find_col(df: pd.DataFrame, candidates: list) -> str:
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        for cand in candidates:
            for col in cols:
                if cand in col:
                    return cols[col]
        return ''

    df = ticket_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    time_col = _find_col(df, ['time', 'date', 'datetime', '交易时间', '成交时间'])
    trader_col = _find_col(df, ['trader', 'name', '交易员'])
    contract_col = _find_col(df, ['contract', '合约', 'ticket'])
    qty_col = _find_col(df, ['qty', 'quantity', '数量'])
    price_col = _find_col(df, ['price', '成交价', '价格'])
    side_col = _find_col(df, ['side', '方向'])

    parsed_ticket = pd.DataFrame()
    if qty_col and contract_col:
        parsed_ticket = pd.DataFrame({
            'timestamp': pd.to_datetime(df[time_col], errors='coerce') if time_col else pd.NaT,
            'trader': df[trader_col].fillna('未填') if trader_col else '未填',
            'contract': df[contract_col].astype(str),
            'quantity': pd.to_numeric(df[qty_col], errors='coerce').fillna(0.0),
            'price': pd.to_numeric(df[price_col], errors='coerce').fillna(0.0) if price_col else 0.0,
        })
        if side_col:
            side_series = df[side_col].astype(str).str.upper()
            sell_mask = side_series.str.contains('S') | side_series.str.contains('卖') | side_series.str.contains('-')
            parsed_ticket.loc[sell_mask, 'quantity'] *= -1
        if time_col:
            parsed_ticket = parsed_ticket[(parsed_ticket['timestamp'].dt.date >= start_date) & (parsed_ticket['timestamp'].dt.date <= end_date)]
    else:
        return {'error': '水单文件缺少必要的合约或数量字段。'}

    system_rows = []
    for log in st.session_state['transaction_log']:
        if log.get('status') != 'active':
            continue
        ts = datetime.fromisoformat(log['date']).date()
        if ts < start_date or ts > end_date:
            continue
        system_rows.append({
            'timestamp': datetime.fromisoformat(log['date']),
            'trader': log['trader'],
            'contract': log['contract'],
            'quantity': log['quantity'],
            'price': log['price'],
        })
    system_df = pd.DataFrame(system_rows)

    def _aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=['trader', 'contract', '净数量', '加权价格', '来源'])
        grouped = df.groupby(['trader', 'contract'], dropna=False)
        rows = []
        for (trader, contract), sub in grouped:
            qty_sum = sub['quantity'].sum()
            weighted_price = (sub['price'] * sub['quantity']).sum() / qty_sum if abs(qty_sum) > 1e-9 else sub['price'].mean()
            rows.append({
                'trader': trader,
                'contract': contract,
                '净数量': qty_sum,
                '加权价格': weighted_price,
                '来源': label,
            })
        return pd.DataFrame(rows)

    ticket_agg = _aggregate(parsed_ticket, '水单')
    system_agg = _aggregate(system_df, '系统')

    merged = pd.merge(system_agg, ticket_agg, on=['trader', 'contract'], how='outer', suffixes=('_系统', '_水单')).fillna(0)
    merged['数量差异'] = merged['净数量_系统'] - merged['净数量_水单']
    merged['价格差异'] = merged['加权价格_系统'] - merged['加权价格_水单']
    corrections = merged[(merged['数量差异'].abs() > 1e-6) | (merged['价格差异'].abs() > 1e-6)]

    return {
        'system': system_df,
        'ticket': parsed_ticket,
        'comparison': merged,
        'corrections': corrections,
        'error': '' if not corrections.empty else '对账完成，未发现差异。'
    }


def build_portfolio_brief() -> str:
    """Compose a compact text summary for AI分析."""
    realised_pl = st.session_state['settings'].get('initial_realised_pl', 0.0) + sum([h['realised_pl'] for h in st.session_state['history']])
    unrealised_pl = 0.0
    exposure_lines = []
    for pos in st.session_state['positions']:
        product = pos['product']
        avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
        current_price = st.session_state['market_prices'].get(pos['contract'], avg_price)
        gross_pl = (current_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
        fee_per_unit = st.session_state['settings']['fees']['brent_per_bbl'] if product == 'Brent' else st.session_state['settings']['fees']['hh_per_mmbtu']
        commission = abs(pos['quantity']) * CONTRACT_MULTIPLIERS[product] * fee_per_unit
        unrealised_pl += gross_pl - commission
        exposure_lines.append(f"{pos['trader']} {pos['contract']} {pos['quantity']:.2f}lots avg {avg_price:.2f} current {current_price:.2f}")
    summary = [
        f"已实现盈亏: {realised_pl:.2f} USD",
        f"未实现盈亏: {unrealised_pl:.2f} USD",
        "持仓快照: " + ("; ".join(exposure_lines) if exposure_lines else '暂无持仓')
    ]
    return "\n".join(summary)


def call_deepseek(api_key: str, prompt: str, context: str = '', model: str = DEFAULT_DEEPSEEK_MODEL) -> str:
    """Call DeepSeek chat API with provided prompt and context."""
    if not api_key:
        return '请先输入 DeepSeek API Key。'
    messages = [
        {"role": "system", "content": "You are a bilingual trading desk analyst. Provide concise risk-aware insights."},
        {"role": "user", "content": f"背景:\n{context}\n\n问题:\n{prompt}"}
    ]
    try:
        resp = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}'},
            json={
                'model': model,
                'messages': messages,
                'temperature': 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get('choices', [{}])[0].get('message', {}).get('content', '未获得返回内容。')
    except requests.RequestException as exc:  # pragma: no cover - runtime API call
        return f'调用 DeepSeek 失败: {exc}'


# ---------------------------- Streamlit UI -----------------------------

# ... 之前的导入和函数定义保持不变 ...

def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(page_title='合约交易分析终端', layout='wide', page_icon='📈')
    init_session_state()

    # Custom CSS with futuristic trading-floor vibe
    st.markdown(
        """
        <style>
        :root {
            --bg: #070b1a;
            --panel: rgba(18, 26, 49, 0.9);
            --accent: #46c6ff;
            --accent-2: #9f7aea;
            --grid: rgba(255,255,255,0.04);
        }
        
        /* 更温和的背景设置 */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            background-attachment: fixed;
        }
        
        .panel {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(70,198,255,0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
            position: relative;
            z-index: 1;
        }
        
        /* 添加轻微的网格效果但不覆盖内容 */
        .panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: 
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(0deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 40px 40px;
            opacity: 0.3;
            border-radius: 12px;
            pointer-events: none;
            z-index: -1;
        }
        
        .panel h2, .panel h3 {
            color: var(--accent);
            background: linear-gradient(90deg, var(--accent), #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .glass-row {
            background: rgba(255,255,255,0.05); 
            padding: 0.75rem 1rem; 
            border-radius: 10px; 
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 0.5rem;
        }
        
        .ticker {
            padding: 0.8rem 1.2rem;
            border-radius: 10px;
            background: linear-gradient(90deg, rgba(70,198,255,0.15), rgba(159,122,234,0.15));
            border: 1px solid rgba(70,198,255,0.3);
            box-shadow: 0 0 20px rgba(70,198,255,0.1);
            font-weight: 600;
            color: #e2e8f0;
            margin: 1rem 0;
            backdrop-filter: blur(5px);
        }
        
        .pill {
            padding: 4px 12px; 
            border-radius: 20px; 
            margin-right: 8px; 
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
            display: inline-block;
        }
        
        .success {color: #34d399;}
        .warning {color: #fbbf24;}
        .danger {color: #f87171;}
        
        .code-card {
            background: rgba(15, 23, 42, 0.8); 
            border-radius: 10px; 
            padding: 1rem; 
            border: 1px solid rgba(255,255,255,0.1);
            margin-top: 1rem;
        }
        
        /* 确保表格可读 */
        .stDataFrame {
            background: rgba(15, 23, 42, 0.6) !important;
        }
        
        table.dataframe tbody tr:nth-child(even) {
            background-color: rgba(30, 41, 59, 0.4) !important;
        }
        
        table.dataframe tbody tr:nth-child(odd) {
            background-color: rgba(15, 23, 42, 0.4) !important;
        }
        
        table.dataframe thead tr {
            background-color: rgba(70,198,255,0.15) !important;
        }
        
        /* 确保文本颜色可读 */
        .stMarkdown, .stText, .stCaption, .stDataFrame {
            color: #e2e8f0 !important;
        }
        
        /* 确保输入框可见 */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div {
            background-color: rgba(30, 41, 59, 0.7) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(70,198,255,0.3) !important;
        }
        
        /* 确保按钮可见 */
        .stButton > button {
            background: linear-gradient(90deg, rgba(70,198,255,0.2), rgba(159,122,234,0.2)) !important;
            border: 1px solid rgba(70,198,255,0.4) !important;
            color: #e2e8f0 !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, rgba(70,198,255,0.3), rgba(159,122,234,0.3)) !important;
            border: 1px solid rgba(70,198,255,0.6) !important;
        }
        
        /* 确保指标卡片可读 */
        .stMetric {
            background: rgba(30, 41, 59, 0.6) !important;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* 确保分割线可见 */
        hr {
            border-color: rgba(70,198,255,0.2) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # 标题
    st.title('合约交易分析终端 v6.0 — Neon Trade Edition')
    st.caption('科技感 UI + 水单自动对账 + DeepSeek 风控洞察')

    # Dashboard metrics
    total_positions = sum(abs(pos['quantity']) for pos in st.session_state['positions'])
    realised_pl = st.session_state['settings'].get('initial_realised_pl', 0.0) + sum([h['realised_pl'] for h in st.session_state['history']])
    unrealised_pl = 0.0
    for pos in st.session_state['positions']:
        product = pos['product']
        avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
        current_price = st.session_state['market_prices'].get(pos['contract'], avg_price)
        gross_pl = (current_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
        fee_per_unit = st.session_state['settings']['fees']['brent_per_bbl'] if product == 'Brent' else st.session_state['settings']['fees']['hh_per_mmbtu']
        commission = abs(pos['quantity']) * CONTRACT_MULTIPLIERS[product] * fee_per_unit
        unrealised_pl += gross_pl - commission
    
    # ... 从这里开始，保持你原来的UI代码不变 ...
    # 原来的代码从这里开始继续...
    metric_cols = st.columns(4)
    metric_cols[0].metric('持仓合约数', len(st.session_state['positions']))
    metric_cols[1].metric('合计手数', f"{total_positions:.3f}")
    metric_cols[2].metric('已实现盈亏 (USD)', f"{realised_pl:.2f}", delta=None)
    metric_cols[3].metric('未实现盈亏 (USD)', f"{unrealised_pl:.2f}", delta_color='inverse' if unrealised_pl < 0 else 'normal')

    # Market pulse ticker
    pulse_lines = []
    for pos in st.session_state['positions']:
        avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
        pulse_lines.append(f"{pos['trader']}·{pos['contract']} {pos['quantity']:.2f} @ {format_price(avg_price, pos['product'])}")
    pulse_text = ' | '.join(pulse_lines) if pulse_lines else '暂无持仓，等待市场信号。'
    st.markdown(f"<div class='ticker'>📡 市场脉冲：{pulse_text}</div>", unsafe_allow_html=True)

    # Layout: two columns
    left_col, right_col = st.columns([1, 2], gap='large')

    # ------------------ Left Column: Controls & Input ------------------
    with left_col:
        # Trade entry panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('记录新交易')
        # Batch import button opens modal form
        if st.button('📥 智能文本批量导入', key='open_batch_import', help='粘贴多条交易记录并批量录入'):
            st.session_state['show_batch_import'] = True

        st.markdown('---', unsafe_allow_html=True)
        with st.form('trade_entry_form', clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                trader = st.selectbox('交易员', TRADERS, index=TRADERS.index(st.session_state['last_selected_trader']))
            with col2:
                product = st.selectbox('品种', list(CONTRACTS.keys()))
            contract = st.selectbox('合约', CONTRACTS[product])
            trade_type = st.selectbox('交易类型', [('regular', '常规交易 (计入盈亏)'), ('adjustment', '成本调整 (优化成本)')], format_func=lambda x: x[1])[0]
            quantity = st.number_input('数量 (负数为卖出)', value=0.0, step=0.001, format='%0.3f')
            price = st.number_input('成交价格', value=0.0, format='%0.4f' if product == 'Henry Hub' else '%0.2f')
            submit = st.form_submit_button('提交交易', type='primary')
            if submit:
                if quantity == 0 or price <= 0:
                    st.warning('请输入有效的数量和价格。')
                else:
                    st.session_state['last_selected_trader'] = trader
                    add_transaction(trader, product, contract, quantity, price, trade_type)
                    st.success('交易已录入。')

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Settings panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('全局费用设置')
        fees_col1, fees_col2 = st.columns(2)
        with fees_col1:
            brent_fee = st.number_input('Brent 费用 (per barrel)', value=st.session_state['settings']['fees']['brent_per_bbl'], step=0.01, format='%0.2f')
        with fees_col2:
            hh_fee = st.number_input('Henry Hub 费用 (per MMBtu)', value=st.session_state['settings']['fees']['hh_per_mmbtu'], step=0.0001, format='%0.4f')
        ex_rate = st.number_input('通用汇率 (USD to RMB)', value=st.session_state['settings']['exchange_rate_rmb'], step=0.01, format='%0.2f')
        init_pl = st.number_input('期初实现盈亏 (USD)', value=st.session_state['settings']['initial_realised_pl'], step=0.01)
        if st.button('保存设置', key='save_settings'):
            st.session_state['settings']['fees']['brent_per_bbl'] = float(brent_fee)
            st.session_state['settings']['fees']['hh_per_mmbtu'] = float(hh_fee)
            st.session_state['settings']['exchange_rate_rmb'] = float(ex_rate)
            st.session_state['settings']['initial_realised_pl'] = float(init_pl)
            rebuild_state_from_logs()
            st.success('设置已保存并重新计算。')
        st.markdown('</div>', unsafe_allow_html=True)

        # Scenario analysis panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('情景分析 / 压力测试')
        delta_brent = st.number_input('Brent 价格变动', value=0.0, format='%0.2f', step=0.01)
        delta_hh = st.number_input('Henry Hub 价格变动', value=0.0, format='%0.4f', step=0.0001)
        if st.button('计算影响', key='run_stress'):
            pl_change, new_total = scenario_analysis(delta_brent, delta_hh)
            st.metric('预估P/L变动', f"{pl_change:.2f}")
            st.metric('预估新总浮动净盈亏', f"{new_total:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Data management & reports panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('数据管理 & 日报')
        # Export data as JSON
        st.download_button('导出全部数据 (JSON)', data=export_json(), file_name=f"trade_data_export_{datetime.now(UTC).isoformat()}.json", mime='application/json', key='export_json')
        # Import data JSON
        json_file = st.file_uploader('导入数据 (JSON)', type=['json'], key='import_data')
        if json_file is not None:
            content = json_file.getvalue().decode('utf-8')
            if import_json(content):
                st.success('数据导入成功。')
            else:
                st.error('数据导入失败，文件格式可能不正确。')
        # Import MTM data
        mtm_file = st.file_uploader('导入行情数据 (JSON)', type=['json'], key='import_mtm')
        if mtm_file is not None:
            content = mtm_file.getvalue().decode('utf-8')
            updated = import_mtm_json(content)
            if updated > 0:
                st.success(f'成功更新 {updated} 个价格。')
                rebuild_state_from_logs()
            else:
                st.error('行情数据导入失败。')
        # Export ledger
        st.download_button('导出逐日台账 (CSV)', data=export_history_csv(), file_name='trade_history.csv', mime='text/csv', key='export_history')
        # Export positions
        st.download_button('导出持仓 (CSV)', data=export_positions_csv(), file_name='positions.csv', mime='text/csv', key='export_positions')
        # Export log
        st.download_button('导出交易日志 (CSV)', data=export_log_csv(), file_name='transaction_log.csv', mime='text/csv', key='export_log')
        # Daily report summary – simply show realised and unrealised totals
        if st.button('生成今日日报摘要', key='daily_report'):
            realised_pl = st.session_state['settings'].get('initial_realised_pl', 0.0) + sum([h['realised_pl'] for h in st.session_state['history']])
            unrealised_pl = 0.0
            for pos in st.session_state['positions']:
                product = pos['product']
                avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
                current_price = st.session_state['market_prices'].get(pos['contract'], avg_price)
                gross_pl = (current_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
                fee_per_unit = st.session_state['settings']['fees']['brent_per_bbl'] if product == 'Brent' else st.session_state['settings']['fees']['hh_per_mmbtu']
                commission = abs(pos['quantity']) * CONTRACT_MULTIPLIERS[product] * fee_per_unit
                unrealised_pl += gross_pl - commission
            report_text = f"今天的总结\n================\n\n累计实现盈亏: {realised_pl:.2f} USD\n当前未实现盈亏: {unrealised_pl:.2f} USD\n\n持仓一览:\n"
            for pos in st.session_state['positions']:
                avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
                report_text += f"{pos['trader']} – {pos['contract']} – {pos['quantity']:.3f} @ {format_price(avg_price, pos['product'])}\n"
            st.text_area('日报摘要', report_text, height=200)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ Right Column: Data & Analysis -------------------
    with right_col:
        # Positions panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('当前持仓')
        search_pos = st.text_input('搜索合约/交易员...', key='search_positions')
        # Build DataFrame for positions
        pos_rows = []
        grand_total_pl = 0.0
        for pos in st.session_state['positions']:
            product = pos['product']
            avg_price = pos['total_value'] / pos['quantity'] if abs(pos['quantity']) > 1e-12 else 0.0
            current_price = st.session_state['market_prices'].get(pos['contract'], avg_price)
            gross_pl = (current_price * pos['quantity'] * CONTRACT_MULTIPLIERS[product]) - (pos['total_value'] * CONTRACT_MULTIPLIERS[product])
            fee_per_unit = st.session_state['settings']['fees']['brent_per_bbl'] if product == 'Brent' else st.session_state['settings']['fees']['hh_per_mmbtu']
            commission = abs(pos['quantity']) * CONTRACT_MULTIPLIERS[product] * fee_per_unit
            floating_pl = gross_pl - commission
            grand_total_pl += floating_pl
            rmb = st.session_state['settings']['exchange_rate_rmb'] or 7.13
            landed_price = 0.0
            if product == 'Brent':
                landed_price = (avg_price * 0.134 + 0.46) * rmb / 28.3
            elif product == 'Henry Hub':
                landed_price = (avg_price * 1.15 + 4.5) * rmb / 28.3
            pos_rows.append({
                '交易员': pos['trader'],
                '合约': pos['contract'],
                '数量': pos['quantity'],
                '均价': avg_price,
                'MTM价格': current_price,
                '浮动净P/L': floating_pl,
                '对应到岸价': landed_price,
                'product': product
            })
        pos_df = pd.DataFrame(pos_rows)
        if search_pos:
            mask = pos_df.apply(lambda row: search_pos.lower() in str(row['合约']).lower() or search_pos.lower() in str(row['交易员']).lower(), axis=1)
            pos_df = pos_df[mask]
        if not pos_df.empty:
            # Format values
            pos_df['数量'] = pos_df['数量'].apply(lambda x: f"{x:.3f}")
            pos_df['均价'] = pos_df.apply(lambda row: format_price(row['均价'], row['product']), axis=1)
            pos_df['MTM价格'] = pos_df.apply(lambda row: format_price(row['MTM价格'], row['product']), axis=1)
            pos_df['浮动净P/L'] = pos_df['浮动净P/L'].apply(lambda x: f"{x:.2f}")
            pos_df['对应到岸价'] = pos_df['对应到岸价'].apply(lambda x: f"{x:.4f}" if x > 0 else '')
            display_df = pos_df.drop(columns=['product'])
            st.dataframe(display_df, width='stretch')
        else:
            st.info('暂无持仓。')
        st.markdown(f"**总浮动净P/L: {'{:.2f}'.format(grand_total_pl)}**")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('交易日志')
        search_log = st.text_input('搜索合约/交易员...', key='search_log')
        # Build list of logs
        logs = []
        for log in sorted(st.session_state['transaction_log'], key=lambda x: x['date'], reverse=True):
            logs.append({
                'id': log['id'],
                '时间': datetime.fromisoformat(log['date']).strftime('%Y-%m-%d %H:%M:%S'),
                '交易员': log['trader'],
                '合约': log['contract'],
                '数量': log['quantity'],
                '价格': log['price'],
                '状态': '已撤销' if log.get('status') == 'reversed' else '有效',
                'product': log['product']
            })
        if search_log:
            logs = [row for row in logs if search_log.lower() in row['合约'].lower() or search_log.lower() in row['交易员'].lower()]
        if logs:
            # Display header row using columns for alignment
            header_cols = st.columns([2, 1, 1, 1, 1, 1])
            header_cols[0].markdown('**时间**')
            header_cols[1].markdown('**交易员**')
            header_cols[2].markdown('**合约**')
            header_cols[3].markdown('**数量**')
            header_cols[4].markdown('**价格**')
            header_cols[5].markdown('**操作**')
            for row in logs:
                cols = st.columns([2, 1, 1, 1, 1, 1])
                cols[0].write(row['时间'])
                cols[1].write(row['交易员'])
                cols[2].write(row['合约'])
                # coloured quantity
                qty_html = f"<span style='color:{'green' if row['数量']>0 else 'red'}'>{row['数量']:.3f}</span>"
                cols[3].markdown(qty_html, unsafe_allow_html=True)
                cols[4].write(format_price(row['价格'], row['product']))
                if row['状态'] == '有效':
                    if cols[5].button('撤销', key=f'reverse_{row["id"]}'):
                        reverse_transaction(row['id'])
                        st.experimental_rerun()
                else:
                    cols[5].markdown("<span style='color:#94a3b8'>已撤销</span>", unsafe_allow_html=True)
        else:
            st.info('暂无交易记录。')
        st.markdown('</div>', unsafe_allow_html=True)

        # History panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('历史平仓记录')
        search_hist = st.text_input('搜索合约/交易员...', key='search_history')
        hist_rows = []
        total_realised_pl = st.session_state['settings'].get('initial_realised_pl', 0.0)
        for hist in sorted(st.session_state['history'], key=lambda x: x['date'], reverse=True):
            total_realised_pl += hist['realised_pl']
            hist_rows.append({
                '日期': hist['date'][:10],
                '交易员': hist['trader'],
                '合约': hist['contract'],
                '平仓量': hist['closed_quantity'],
                '开仓价': hist['open_price'],
                '平仓价': hist['close_price'],
                '实现净P/L': hist['realised_pl'],
                'product': hist['product']
            })
        hist_df = pd.DataFrame(hist_rows)
        if search_hist:
            mask = hist_df.apply(lambda row: search_hist.lower() in str(row['合约']).lower() or search_hist.lower() in str(row['交易员']).lower(), axis=1)
            hist_df = hist_df[mask]
        if not hist_df.empty:
            display_df = hist_df.copy()
            display_df['平仓量'] = display_df['平仓量'].apply(lambda x: f"{x:.3f}")
            display_df['开仓价'] = display_df.apply(lambda row: format_price(row['开仓价'], row['product']), axis=1)
            display_df['平仓价'] = display_df.apply(lambda row: format_price(row['平仓价'], row['product']), axis=1)
            display_df['实现净P/L'] = display_df['实现净P/L'].apply(lambda x: f"{x:.2f}")
            display_df = display_df.drop(columns=['product'])
            st.dataframe(display_df, width='stretch')
            st.markdown(f"**累计实现盈亏: {'{:.2f}'.format(total_realised_pl)} USD**")
        else:
            st.info('暂无平仓记录。')
        st.markdown('</div>', unsafe_allow_html=True)

        # Infographics panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('Infographics 数据分析')
        pie_chart, pl_chart = build_infographics()
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.altair_chart(pie_chart, width='stretch')
        with chart_col2:
            st.altair_chart(pl_chart, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        # Ticket reconciliation panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('水单自动对账 (Beta)')
        today = datetime.now().date()
        default_range = (today - timedelta(days=7), today)
        date_range = st.date_input('选择对账时间范围', value=default_range)
        start_date, end_date = (date_range[0], date_range[1]) if isinstance(date_range, (list, tuple)) else (today, today)
        ticket_file = st.file_uploader('上传水单明细 (CSV / XLSX)', type=['csv', 'xlsx'], key='ticket_upload')
        template_img = st.file_uploader('上传水单模板/截图 (可选)', type=['png', 'jpg', 'jpeg'], key='ticket_image')
        if template_img is not None:
            st.image(template_img, caption='水单模板预览', use_column_width=True)
        if st.button('执行对账', key='run_reconciliation'):
            ticket_df, err = load_ticket_data(ticket_file)
            if err:
                st.error(err)
            else:
                with st.spinner('对账中，请稍候...'):
                    recon = reconcile_tickets(ticket_df, start_date, end_date)
                st.session_state['recon_result'] = recon
        recon_result = st.session_state.get('recon_result')
        if recon_result:
            if recon_result.get('error'):
                st.info(recon_result['error'])
            if not recon_result.get('system', pd.DataFrame()).empty:
                st.markdown('**系统交易 (筛选后)**')
                st.dataframe(recon_result['system'], width='stretch')
            if not recon_result.get('ticket', pd.DataFrame()).empty:
                st.markdown('**水单明细 (筛选后)**')
                st.dataframe(recon_result['ticket'], width='stretch')
            if not recon_result.get('comparison', pd.DataFrame()).empty:
                st.markdown('**合约差异对比**')
                st.dataframe(recon_result['comparison'], width='stretch')
                st.download_button('导出差异 (CSV)', recon_result['comparison'].to_csv(index=False), file_name='reconciliation_delta.csv', mime='text/csv')
            if recon_result.get('corrections') is not None and not recon_result['corrections'].empty:
                st.markdown('**需纠错条目**')
                st.dataframe(recon_result['corrections'], width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        # DeepSeek analysis panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('AI 深度洞察 (DeepSeek)')
        st.caption('快速让 AI 复核持仓、发现风险或生成简报。')
        api_key = st.text_input('DeepSeek API Key', value=st.session_state.get('deepseek_key', ''), type='password')
        st.session_state['deepseek_key'] = api_key
        prompt = st.text_area('提出你的问题或分析需求', value='请审视近期持仓风险，给出两条调整建议。')
        include_brief = st.checkbox('自动附加持仓摘要', value=True)
        if st.button('生成 AI 反馈', key='run_deepseek'):
            context = build_portfolio_brief() if include_brief else ''
            with st.spinner('调用 DeepSeek...'):
                reply = call_deepseek(api_key, prompt, context)
            st.session_state['deepseek_reply'] = reply
        if st.session_state.get('deepseek_reply'):
            st.markdown('**DeepSeek 反馈**')
            reply_html = st.session_state['deepseek_reply'].replace('\n', '<br>')
            st.markdown(f"<div class='code-card'>{reply_html}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ Batch Import Modal ------------------
    # Show modal if triggered
    if st.session_state.get('show_batch_import', False):
        with st.modal('智能文本批量导入'):
            st.write('请粘贴您的交易记录文本。每行一条交易。')
            st.caption('示例1: You bot 5x/m mar-Dec brt at 61.16 otc\n示例2: 61.43 61.22 ... (直接换行跟随明细价格)')
            text_input = st.text_area('在此粘贴交易文本...', key='batch_input')
            if st.button('解析预览', key='parse_import'):
                parsed = parse_batch_input(text_input or '', st.session_state['last_selected_trader'])
                st.session_state['parsed_trades_buffer'] = parsed
            parsed_trades = st.session_state.get('parsed_trades_buffer', [])
            if parsed_trades:
                valid_count = sum(1 for t in parsed_trades if t['is_valid'])
                st.markdown(f"解析完成：共 {len(parsed_trades)} 笔条目，有效 {valid_count} 笔。")
                # Build preview table
                preview_rows = []
                for t in parsed_trades:
                    preview_rows.append({
                        '状态': '有效' if t['is_valid'] else '无效',
                        '交易员': t['trader'] or '-',
                        '品种': t['product'] or '-',
                        '合约': t['contract'] or '-',
                        '方向': '买入' if t['side'] == 1 else '卖出',
                        '数量': f"{t['qty']:.3f}" if t['qty'] else '-',
                        '价格': f"{t['price']}" if t['price'] else '-',
                    })
                preview_df = pd.DataFrame(preview_rows)
                st.dataframe(preview_df, width='stretch')
                # Confirmation buttons
                if st.button('确认提交', disabled=(valid_count == 0), key='confirm_batch_submit'):
                    for t in parsed_trades:
                        if t['is_valid']:
                            add_transaction(t['trader'], t['product'], t['contract'], t['final_qty'], t['price'], 'regular')
                    st.success(f'成功导入 {valid_count} 条交易记录。')
                    st.session_state['parsed_trades_buffer'] = []
                    st.session_state['show_batch_import'] = False
                    st.experimental_rerun()
                if st.button('取消', key='cancel_batch_import'):
                    st.session_state['parsed_trades_buffer'] = []
                    st.session_state['show_batch_import'] = False
            else:
                st.info('粘贴文本后点击"解析预览"以预览交易。')


if __name__ == '__main__':
    main()
