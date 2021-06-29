import pandas as pd
import sweetviz as sv
import os

overriden = sv.config_parser.read("override.ini")
print(overriden) # Should not be empty

filename='./data/data_ldc_vijay/vijay_ldc_paper_data.csv'
df=pd.read_csv(filename,  delimiter=';', index_col='Training')
df.drop(labels=['Stream'], axis=1, inplace=True)

col_names=[
            'B(m)     - Channel width                        ', 
            'H(m)     - Cross-sectional average flow depth   ', 
            'U(m/s)   - Cross-sectional average flow velocity', 
            'u*(m/s)  - Shear velocity                       ', 
            'Q(m3/s)  - Flow discharge                       ', 
            'U/u*     - Relative shear velocity              ', 
            'Beta     - Channel shape parameter              ', 
            'Sigma    - Channel sinuosity                    ', 
            'Kx(m2/s) - Longitudinal dispersion coefficient  ',
           ]

df.columns    = col_names
target_names  = ['$K_x$']

df_train =    df[df.index=='*']    
df_test  =    df[df.index=='**']    
df_train.describe().T.to_latex('/tmp/train.tex')
df_test.describe().T.to_latex('/tmp/test.tex')

advert_report = sv.analyze(df)
#advert_report.show_html('ldc_dataset.html')

sv.feature_config.FeatureConfig(force_num='3')

df1 = sv.compare([df_train, "Training Data"], 
                 [df_test, "Test Data"],
                 #target_feat='Kx(m2/s) - Longitudinal dispersion coefficient  ',
                 )
fn='ldc_train_test_compare.html'
df1.show_html(fn)
os.system('cp '+fn+' /tmp/')

