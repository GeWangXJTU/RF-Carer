from sklearn.preprocessing import MinMaxScaler


def mapminmax(data, min_val, max_val):
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()


