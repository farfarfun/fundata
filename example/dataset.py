from notedata.dataset import CriteoData, ElectronicsData

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
notebuild build
cd example

pip install git+https://gitee.com/notechats/notedata.git

/root/anaconda3/bin/python /root/workspace/notechats/notedata/example/dataset.py

"""
