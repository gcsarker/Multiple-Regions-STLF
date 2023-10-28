import numpy as np

# IQR method of outlier detection
def find_outliers_IQR(df_):
  q1=df_.quantile(0.25)
  q3=df_.quantile(0.75)
  IQR=q3-q1
  outliers = df_[((df_<(q1-1.5*IQR)) | (df_>(q3+1.5*IQR)))]
  return np.array(outliers.index), q1-1.5*IQR, q3+1.5*IQR