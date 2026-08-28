from fundata.dataset import CriteoData, ElectronicsData

electronic = ElectronicsData()
# electronic.download()

criteo = CriteoData()
criteo.download()
print(criteo.criteo_sample)
print(criteo.build_dataset())
# electronic.download_raw_0()
# get_electronics(dataset=data)
# get_movielens(dataset=data)
# get_adult_data(data)
# get_porto_seguro_data(data)
# get_bitly_usagov_data(data)


"""
cd ..
funbuild build
cd example

pip install fundata

/root/anaconda3/bin/python /root/workspace/farfarfun/fundata/example/dataset.py

"""
