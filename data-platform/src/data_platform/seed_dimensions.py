import itertools

import pandas as pd
from data_platform.db import get_engine


def seed_dim_date(start='2024-01-01', end='2027-12-31'):
    engine = get_engine()
    dates = pd.date_range(start=start, end=end, freq='D')
    rows = []
    for d in dates:
        rows.append({
            'date_key':     int(d.strftime('%Y%m%d')),
            'full_date':    d.date(),
            'year':         d.year,
            'quarter':      d.quarter,
            'month':        d.month,
            'month_name':   d.strftime('%B'),
            'day_of_month': d.day,
            'day_of_week':  d.dayofweek,
            'is_month_end': bool(d.is_month_end)
        })
    df = pd.DataFrame(rows)
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(__import__('sqlalchemy').text("""
                INSERT INTO gold.dim_date
                VALUES (:date_key,:full_date,:year,:quarter,:month,
                        :month_name,:day_of_month,:day_of_week,:is_month_end)
                ON CONFLICT DO NOTHING
            """), row.to_dict())
    print(f"seeded dim_date: {len(rows)} rows")

def seed_dim_income_band():
    engine = get_engine()
    rows = [
        ('unknown',    None,  None,  True,  0),
        ('0-2000',     0,     2000,  False, 1),
        ('2000-4000',  2000,  4000,  False, 2),
        ('4000-6000',  4000,  6000,  False, 3),
        ('6000-10000', 6000,  10000, False, 4),
        ('10000+',     10000, None,  False, 5),
    ]
    with engine.begin() as conn:
        for row in rows:
            conn.execute(__import__('sqlalchemy').text("""
                INSERT INTO gold.dim_income_band
                    (band_label, lower_bound, upper_bound, is_unknown, sort_order)
                VALUES (:label, :lower, :upper, :unknown, :sort)
                ON CONFLICT DO NOTHING
            """), {'label': row[0], 'lower': row[1], 'upper': row[2],
                   'unknown': row[3], 'sort': row[4]})
    print(f"seeded dim_income_band: {len(rows)} rows")

def seed_dim_utilisation_band():
    engine = get_engine()
    rows = [
        ('0-0.1',          0,    0.1,  False, 1),
        ('0.1-0.3',        0.1,  0.3,  False, 2),
        ('0.3-0.6',        0.3,  0.6,  False, 3),
        ('0.6-1.0',        0.6,  1.0,  False, 4),
        ('>1.0 anomalous', 1.0,  None, True,  5),
    ]
    with engine.begin() as conn:
        for row in rows:
            conn.execute(__import__('sqlalchemy').text("""
                INSERT INTO gold.dim_utilisation_band
                    (band_label, lower_bound, upper_bound, is_anomalous, sort_order)
                VALUES (:label, :lower, :upper, :anomalous, :sort)
                ON CONFLICT DO NOTHING
            """), {'label': row[0], 'lower': row[1], 'upper': row[2],
                   'anomalous': row[3], 'sort': row[4]})
    print(f"seeded dim_utilisation_band: {len(rows)} rows")

def seed_dim_delinquency_profile():
    engine = get_engine()
    combos = list(itertools.product([False, True], repeat=4))
    with engine.begin() as conn:
        for combo in combos:
            has_30_59, has_60_89, has_90_plus, has_sentinel = combo
            if has_sentinel:
                severity = 'unreliable'
            elif has_90_plus:
                severity = 'severe'
            elif has_60_89:
                severity = 'moderate'
            elif has_30_59:
                severity = 'mild'
            else:
                severity = 'none'
            conn.execute(__import__('sqlalchemy').text("""
                INSERT INTO gold.dim_delinquency_profile
                    (has_30_59_late, has_60_89_late, has_90_plus_late,
                     has_sentinel_code, severity_label)
                VALUES (:a, :b, :c, :d, :sev)
                ON CONFLICT DO NOTHING
            """), {'a': has_30_59, 'b': has_60_89, 'c': has_90_plus,
                   'd': has_sentinel, 'sev': severity})
    print(f"seeded dim_delinquency_profile: {len(combos)} rows")

def main():
    seed_dim_date()
    seed_dim_income_band()
    seed_dim_utilisation_band()
    seed_dim_delinquency_profile()


if __name__ == '__main__':
    main()