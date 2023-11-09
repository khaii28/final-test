# %%
import warnings

warnings.simplefilter(action= 'ignore', category = FutureWarning)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

# %% [markdown]
# Menggabungkan dataset menjadi satu

# %%
link_penumpang_bus_sekolah = {
    '2017' : {
            'januari': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/e4b640be-d9a2-4ea0-b94f-b5313385c04c/download/Penumpang-Bus-Sekolah-Januari-2017.csv',
            'februari': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/37205a79-ddae-4ef1-a3ac-4df1fb343754/download/Penumpang-Bus-Sekolah-Februari-2017.csv',
            'maret': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/24f1c29f-539e-442a-9a63-c68fff3093da/download/Penumpang-Bus-Sekolah-Maret-2017.csv',
            'april': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/fb5b4760-2a36-47ef-b037-e99c23fd9461/download/Penumpang-Bus-Sekolah-April-2017.csv',
            'mei': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/f9236311-cdec-438d-9127-bf426a0b0b42/download/Penumpang-Bus-Sekolah-Mei-2017.csv',
            'juni': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/724e0d36-a683-40e2-aa5e-c4310e873b5d/download/Penumpang-Bus-Sekolah-Juni-2017.csv',
            'juli': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/6b325948-ddf6-4da1-a96f-268055859285/download/Penumpang-Bus-Sekolah-Juli-2017.csv',
            'agustus': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/04781028-0ffa-4ef7-9356-d52460e1f824/download/Penumpang-Bus-Sekolah-Agustus-2017.csv',
            'september': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/18a8356c-5263-4119-be05-652c28b8ea67/download/Penumpang-Bus-Sekolah-September-2017.csv',
            'oktober': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/7eb98893-67c6-403d-83cd-951d6dacf197/download/Penumpang-Bus-Sekolah-Oktober-2017.csv',
            'november': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/2a9e76b7-b044-4384-a329-5bc681bf8cf7/download/Penumpang-Bus-Sekolah-November-2017.csv',
            'desember': 'https://data.jakarta.go.id/dataset/7cf2294c-b2d8-4960-9473-45b973ab4273/resource/29c3eced-52c0-408d-8b65-b6047ad29b22/download/Penumpang-Bus-Sekolah-Desember-2017.csv'
            },
    '2018' : {
            'januari': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/66adb89a-ae5d-462a-b0b4-3029ae108693/download/Data-Penumpang-Bus-Sekolah-Januari-2018.csv',
            'februari': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/f1ad67c2-1a66-4d48-bec4-03d67b3595b4/download/Data-Penumpang-Bus-Sekolah-Februari-2018.csv',
            'maret': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/9db1c6a8-ba55-4099-b325-6f9708dd93be/download/Data-Penumpang-Bus-Sekolah-Maret-2018.csv',
            'april': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/cfe8b2a1-ca56-4f54-9719-2fdef07b3f58/download/Data-Penumpang-Bus-Sekolah-April-2018.csv',
            'mei': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/2f6cfb75-0b15-4c58-8259-8258ba744931/download/Data-Penumpang-Bus-Sekolah-Mei-2018.csv',
            'juni': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/f8b54e46-dd99-4303-9604-00221a05e95d/download/Data-Penumpang-Bus-Sekolah-Juni-2018.csv',
            'juli': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/64b338e7-f40b-4aab-a2d7-0dcd305e4ce8/download/Data-Penumpang-Bus-Sekolah-Juli-2018.csv',
            'agustus': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/935859ce-a92b-4b3a-99fb-f517f4a7b89d/download/Data-Penumpang-Bus-Sekolah-Agustus-2018.csv',
            'september': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/7a281451-6b3a-492f-881c-daab0440b338/download/Data-Penumpang-Bus-Sekolah-September-2018.csv',
            'oktober': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/1871014a-94bb-4693-ace0-230f629d089d/download/Data-Penumpang-Bus-Sekolah-Oktober-2018.csv',
            'november': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/55a21757-3313-44d6-bafe-0721b86b5f99/download/Data-Penumpang-Bus-Sekolah-November-2018.csv',
            'desember': 'https://data.jakarta.go.id/dataset/737d708e-0eea-4734-acdc-9038a38692ca/resource/12040d13-1079-4c4c-b1a8-73ce3f821d17/download/Data-Penumpang-Bus-Sekolah-Desember-2018.csv'
            },
    '2019' : {
            'januari': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/f268abb0-a698-48c7-89c0-98e89db6fde5/download/Data-Penumpang-Bus-Sekolah-Januari-2019.csv',
            'februari': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/2db6ec67-2217-4da1-a994-7bf770502997/download/Data-Penumpang-Bus-Sekolah-Februari-2019.csv',
            'maret': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/2e15072d-c4dd-4334-9b73-bbe09cd05d38/download/Data-Penumpang-Bus-Sekolah-Maret-2019.csv',
            'april': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/cadde2eb-61af-4e66-ad99-2d96daea2730/download/Data-Penumpang-Bus-Sekolah-April-2019.csv',
            'mei': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/d0f8993e-e422-4241-adec-b638ec673087/download/Data-Penumpang-Bus-Sekolah-Mei-2019.csv',
            'juni': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/dfdd18da-f011-419c-ba11-4cac7011f086/download/Data-Penumpang-Bus-Sekolah-Juni-2019.csv',
            'juli': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/17a1f125-3cdd-43f9-ade5-59265a8a629b/download/Data-Penumpang-Bus-Sekolah-Juli-2019.csv',
            'agustus': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/8b581cad-7b58-4761-b1fa-aa677a4323ef/download/Data-Penumpang-Bus-Sekolah-Agustus-2019.csv',
            'september': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/a7cf979a-520b-40e1-a2dc-ef131f350379/download/Data-Penumpang-Bus-Sekolah-September-2019.csv',
            'oktober': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/42444774-1bea-4e56-b626-d843f73580ea/download/Data-Penumpang-Bus-Sekolah-Oktober-2019.csv',
            'november': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/ab104258-0574-4b62-b4cc-63843d8b843d/download/Data-Penumpang-Bus-Sekolah-November-2019.csv',
            'desember': 'https://data.jakarta.go.id/dataset/97ce3df4-c76c-41f5-a26f-6661fd760701/resource/6f307bfe-836d-4bd9-9f96-1f667925ab8c/download/Data-Penumpang-Bus-Sekolah-Desember-2019.csv'
            }
    }

# %%
df_penumpang_bus_sekolah_all = pd.DataFrame()
for tahun, nested_dict in link_penumpang_bus_sekolah.items():
  for bulan, link in nested_dict.items():
    try:
      df_penumpang_bus_sekolah_temp = pd.read_csv(link)
    except:
      df_penumpang_bus_sekolah_temp = pd.read_csv(link, encoding='latin-1')
    df_penumpang_bus_sekolah_temp['bulan'] = bulan
    df_penumpang_bus_sekolah_temp['tahun'] = int(tahun)

    df_penumpang_bus_sekolah_all = df_penumpang_bus_sekolah_all.append(df_penumpang_bus_sekolah_temp)

print(df_penumpang_bus_sekolah_all.shape)
df_penumpang_bus_sekolah_all.head()


# %% [markdown]
# Manipulation data pada jumlah_bus dan jumlah_penumpang supaya berada dalam satu format type
# 
# Antisipasi agar hasil tidak eror terlebih dahulu memasukan hasil download file csv penumpang bus secara keseluruhan dari tahun 2017-2019, karena keberhasilan proses dibawah ini adalah setelah mengupload file csv tersebut

# %%
import pandas as pd
import glob

combined_dataset = pd.DataFrame()
years = [2017, 2018, 2019]
months = ['januari', 'februari','maret','april','mei','juni','juli','agustus','september','oktober','november','desember']

for year in years:
    for month in months:
        file_pattern = f'../content/*bus-sekolah-{month}-{year}.csv'
        file_names = glob.glob(file_pattern)
        print(file_names)

        for file_name in file_names:

            df = pd.read_csv(file_name, encoding='cp1252')
            split_file_names = file_name.split('-')
            print(split_file_names)
            file_year = int(split_file_names[-1].split('.')[0])
            file_month = split_file_names[-2]

            df['Tahun'] = file_year
            df['Bulan'] = file_month
            df['jumlah_bus'].fillna(0, inplace=True)
            df['jumlah_bus']= df['jumlah_bus'].astype(int)
            df['jumlah_penumpang']= df['jumlah_penumpang'].astype(str)
            df['jumlah_penumpang']= df['jumlah_penumpang'].str.replace(r'[^\d]','', regex=True)
            df['jumlah_penumpang']= pd.to_numeric(df['jumlah_penumpang'], errors='coerce').fillna(0).astype(int)
            df.dropna(axis=1, how='all', inplace=True)

            combined_dataset = pd.concat([combined_dataset, df], ignore_index=True)

print(combined_dataset)



# %% [markdown]
# Agregasi sum, mean, dan max dari jumlah_bus dan jumlah_penumpang berdasarkan tahun

# %%
aggregated_data = combined_dataset.groupby(['Tahun']).agg({
    'jumlah_bus': ['sum', 'mean', 'max'],
    'jumlah_penumpang': ['sum','mean', 'max']
}).reset_index()

print(aggregated_data)

aggregated_data.to_json('file.json')


