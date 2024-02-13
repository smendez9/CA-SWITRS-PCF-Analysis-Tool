# import libraries
import base64
from io import BytesIO
from flask import Flask, request, render_template
import pandas as pd
import joblib
import matplotlib.pyplot as plt

application = Flask(__name__)


# Model Code

@application.route('/')
@application.route('/about')
def about():
    return render_template("about.html")

@application.route('/resume')
def resume():
    return render_template("resume.html")


@application.route('/PrimaryCollisionFactorPredictor', methods=['GET', 'POST'])
def PrimaryCollisionFactorPredictor():
    # create output mapping for easier user readability of predictions
    output_mapping = {
        '0': 'Unknown',
        '1': 'DUI',
        '2': 'Impeding Traffic',
        '3': 'Unsafe Speed',
        '4': 'Following Too Closely',
        '5': 'Wrong Side of Road',
        '6': 'Improper Passing',
        '7': 'Unsafe Lane Change',
        '8': 'Improper Turning',
        '9': 'Automobile Right of Way',
        '10': 'Ped Right of Way',
        '11': 'Ped Violation',
        '12': 'Traffic Signals/Signs',
        '13': 'Hazardous Parking',
        '14': 'Lights',
        '15': 'Brakes',
        '16': 'Other Equiptment',
        '17': 'Other Hazardous Violation',
        '18': 'Unsafe Starting or Backing',
        '19': 'Other Improper Driving',
        '20': 'Not Stated'
    }

    if request.method == 'POST':
        # collect data from form
        PARTY_SOBRIETY = request.form.get('PARTY_SOBRIETY')
        OAF_VIOL_CAT = request.form.get('OAF_VIOL_CAT')
        MOVE_PRE_ACC = request.form.get('MOVE_PRE_ACC')
        DAY_OF_WEEK = request.form.get('DAY_OF_WEEK')
        CHP_BEAT_TYPE = request.form.get('CHP_BEAT_TYPE')
        WEATHER_1 = request.form.get('WEATHER_1')
        PARTY_COUNT = request.form.get('PARTY_COUNT')
        TYPE_OF_COLLISION = request.form.get('TYPE_OF_COLLISION')
        MVIW = request.form.get('MVIW')
        ROAD_COND_1 = request.form.get('ROAD_COND_1')
        ALCOHOL_INVOLVED = request.form.get('ALCOHOL_INVOLVED')
        COLLISION_MONTH = request.form.get('COLLISION_MONTH')
        COLLISION_DAY = request.form.get('COLLISION_DAY')
        COLLISION_HOUR = request.form.get('COLLISION_HOUR')
        COUNTY = request.form.get('COUNTY')

        # select data for model
        data = [PARTY_SOBRIETY, OAF_VIOL_CAT, MOVE_PRE_ACC, DAY_OF_WEEK,
                CHP_BEAT_TYPE, WEATHER_1, PARTY_COUNT,
                TYPE_OF_COLLISION, MVIW, ROAD_COND_1, ALCOHOL_INVOLVED,
                COLLISION_MONTH, COLLISION_DAY, COLLISION_HOUR, COUNTY]

        # to data frame
        data = pd.DataFrame(
            {'PARTY_SOBRIETY': [PARTY_SOBRIETY], 'OAF_VIOL_CAT': [OAF_VIOL_CAT], 'MOVE_PRE_ACC': [MOVE_PRE_ACC],
             'DAY_OF_WEEK': [DAY_OF_WEEK],
             'CHP_BEAT_TYPE': [CHP_BEAT_TYPE], 'WEATHER_1': [WEATHER_1], 'PARTY_COUNT': [PARTY_COUNT],
             'TYPE_OF_COLLISION': [TYPE_OF_COLLISION], 'MVIW': [MVIW], 'ROAD_COND_1': [ROAD_COND_1],
             'ALCOHOL_INVOLVED': [ALCOHOL_INVOLVED],
             'COLLISION_MONTH': [COLLISION_MONTH], 'COLLISION_DAY': [COLLISION_DAY], 'COLLISION_HOUR': [COLLISION_HOUR],
             'COUNTY': [COUNTY]})

        # import model
        file = open('finalmodel.pkl', 'rb')

        model = joblib.load(file)

        # predict PCF
        pred = model.predict(data)[0]

        # view feature importance
        feature_importance = model.feature_importances_

        # select cols for feature importance
        features = ['PARTY_SOBRIETY', 'OAF_VIOL_CAT', 'MOVE_PRE_ACC', 'DAY_OF_WEEK',
                    'CHP_BEAT_TYPE', 'WEATHER_1', 'PARTY_COUNT',
                    'TYPE_OF_COLLISION', 'MVIW', 'ROAD_COND_1', 'ALCOHOL_INVOLVED',
                    'COLLISION_MONTH', 'COLLISION_DAY', 'COLLISION_HOUR', 'COUNTY']

        # sort features by importance
        importance_dict = dict(zip(features, feature_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=False)

        # plot the importance
        plt.figure(figsize=(10, 8))
        plt.barh([x[0] for x in sorted_importance], [x[1] for x in sorted_importance], color='orange')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()

        # save the plot to a BytesIO object to encode it
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # encode the image in base64 and decoding it to UTF-8 to embed in HTML
        plot_url = base64.b64encode(img.getvalue()).decode()

        # clear current figure to free memory after encoding
        plt.clf()

        prediction_key = str(int(pred))
        prediction = output_mapping.get(prediction_key, 'Unknown prediction')  # output prediction as string

        return render_template("result.html", prediction=prediction, plot_url=plot_url)

    else:
        return render_template("PrimaryCollisionFactorPredictor.html")

@application.route('/projects')
def projects():
    return render_template("projects.html")

if __name__ == '__main__':
    application.run()
