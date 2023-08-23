import os
import env
import pandas as pd

from sklearn.model_selection import train_test_split


########################################### main acquire function #######################################################
def get_db_url(db, user, password, host):
    url = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return(url)

def get_telco_data(file_name="telco_churn.csv") -> pd.DataFrame:
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    query = """SELECT *
               FROM customers
               LEFT JOIN contract_types
               USING(contract_type_id)
               LEFT JOIN internet_service_types
               USING (internet_service_type_id)
               LEFT JOIN payment_types
               USING (payment_type_id)"""
    connection = get_db_url("telco_churn", user=env.user, password=env.password, host=env.host)
    df = pd.read_sql(query, connection)
    df.to_csv(file_name, index=False)
    return df

########################################### prep telco function #######################################################

def prep_telco(df):
    df = df.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id'])
    df['internet_service_type'] = df['internet_service_type'].fillna('No internet service')

    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})

    dummy_df = pd.get_dummies(df[['multiple_lines',
                                     'online_security',
                                     'online_backup',
                                     'device_protection',
                                     'tech_support',
                                     'streaming_tv',
                                     'streaming_movies',
                                     'contract_type',
                                     'internet_service_type',
                                     'payment_type']],
                                  drop_first=True).astype(int)

    df = pd.concat( [df, dummy_df], axis=1 )

    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)

    return df

########################################### split function #######################################################


def split_telco_data(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.internet_service_type)
    train, validate = train_test_split(train_validate,
        test_size=.3,
        random_state=123,
        stratify=train_validate.internet_service_type)
    return train, validate, test