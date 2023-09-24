import pandas as pd
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz, process


def filter_transactions(df, year, excluded_vendors):
    return df[(df['Date'].dt.year >= year) &
              (df['Transaction Type'] == 'debit') &
              (df['Category'] != 'Transfer') &
              (~df['Description'].isin(excluded_vendors))]


def refine_description(description, prefix_mapping):
    for prefix, generalized_name in prefix_mapping.items():
        prefix = str(prefix)
        description = str(description)
        if description.startswith(prefix):
            return generalized_name
    return description


def refined_heuristic_grouping(transactions_df, prefix_mapping):
    heuristic_description_groups = defaultdict(list)
    descriptions = transactions_df['Original Description'].unique().tolist()
    processed_descriptions = set()

    for description in descriptions:
        refined_description = refine_description(description, prefix_mapping)

        if refined_description in processed_descriptions:
            continue

        matches = process.extract(refined_description, descriptions, limit=20, scorer=fuzz.token_set_ratio)
        similar_descriptions = [match[0] for match in matches if match[1] > 80]

        similar_indices = transactions_df[
            transactions_df['Original Description'].isin(similar_descriptions)].index.tolist()
        heuristic_description_groups[refined_description].extend(similar_indices)
        processed_descriptions.update(similar_descriptions)

    return heuristic_description_groups


def analyze_recurring_amounts(transactions_df):
    amount_frequencies = Counter(transactions_df['Amount'].round(2))
    recurring_amount_groups = defaultdict(list)

    for amount, count in amount_frequencies.items():
        if count > 1:
            indices = transactions_df[transactions_df['Amount'].round(2) == amount].index.tolist()
            recurring_amount_groups[amount].extend(indices)

    return recurring_amount_groups


def frequency_analysis(dates):
    if len(dates) < 2:
        return 'Irregular'
    dates.sort()
    gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    avg_gap = sum(gaps) / len(gaps)
    if 25 <= avg_gap <= 35 and len(dates) >= 3:
        return 'Monthly'
    elif 6 <= avg_gap <= 8 and len(dates) >= 3:
        return 'Weekly'
    elif 360 <= avg_gap <= 370 and len(dates) >= 2:
        return 'Annually'
    else:
        return 'Irregular'


def main():
    file_path = 'transactions(1).csv'
    df = pd.read_csv(file_path, parse_dates=['Date'])
    excluded_vendors = ['Betterment', 'Chase', 'Citi']
    year = 2019
    filtered_df = filter_transactions(df, year, excluded_vendors)

    prefix_mapping = {
        'AMAZON RETAIL': 'Amazon Retail',
        'AMAZON MARKETPLACE': 'Amazon Marketplace'
    }

    heuristic_description_groups = refined_heuristic_grouping(filtered_df, prefix_mapping)

    final_rows = []
    for group_key, indices in heuristic_description_groups.items():
        group = filtered_df.loc[indices]
        category = group['Category'].mode()[0]  # Most frequent category in the group
        dates = group['Date'].tolist()
        frequency = frequency_analysis(dates)
        final_rows.append({
            'Group Key': group_key,
            'Category': category,
            'Transaction Count': len(indices),
            'Total Amount': group['Amount'].sum(),
            'Frequency': frequency
        })

    final_df = pd.DataFrame(final_rows)
    final_df = final_df[final_df['Transaction Count'] > 1]  # Exclude groups with a single transaction
    final_df = final_df.sort_values(by='Transaction Count',
                                    ascending=False)  # Sort by Transaction Count in descending order

    # Export to CSV
    final_df.to_csv('recurring_transactions.csv', index=False)


if __name__ == "__main__":
    main()
