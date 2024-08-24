import numpy as np
import joblib

from utils import open_nea_table, download_nea_table
from mr_utils import creating_MR_ML_table, creating_R_ML_table
from mr_ml_utils import RF_training_with_display, RF_training


#If needed to update the NEA table then first run  "download_nea_table"
#download_nea_table()
nea_full_table = open_nea_table()

# Creating the final table with R, M, and Teq.
MR_ML_table = creating_MR_ML_table(nea_full_table)
R_ML_table = creating_R_ML_table(nea_full_table)


def MR_RF_training(MR_ML_table):
    # Function for training the random forest model based on MR_ML_table data

    logMR_ML_table = MR_ML_table.copy()
    logMR_ML_table['pl_bmasse_log'] = np.log10(logMR_ML_table['pl_bmasse'])

    y_log = logMR_ML_table['pl_bmasse_log']
    X_log = logMR_ML_table.drop(columns=['pl_bmasse', 'pl_bmasse_log'])

    best_model, final_trained_model_train_data, final_trained_model_full_data = RF_training(X_log, y_log, test_size=0.15, n_trials=100)

    # Save the model and its parameters
    joblib.dump(final_trained_model_full_data, 'RF_model_trained.pkl')

    return final_trained_model_full_data



def mass_from_radius(R_ML_table, training=True):
    # Function for predicting the mass of exoplanets from their radius

    if training:
        final_trained_model_full_data = MR_RF_training(MR_ML_table)
    if not training:
        final_trained_model_full_data = joblib.load('RF_model_trained.pkl')

    M_pred = np.power(10, final_trained_model_full_data.predict(R_ML_table))

    return M_pred

if __name__ == "__main__":
    M_pred = mass_from_radius(R_ML_table, training = False)
    print ('Done')
