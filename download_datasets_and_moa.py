import os.path
import gdown
import zipfile
import shutil

z_files = {
    'covtFD': ['https://drive.google.com/file/d/13MiPqDzuKzv1RvhAqKFCTmCKav3C6s5P/view?usp=share_link', 'zip'],
    'covtype': ['https://drive.google.com/file/d/15kfa2rdnQgwoFVr5MG5hyjqLDNY_v94L/view?usp=share_link', 'zip'],
    'Hyper100k': ['https://drive.google.com/file/d/1tjcKD8kzNfv5RBi7cTymCjoIBFD0ItZV/view?usp=share_link', 'zip'],
    'RBFm_100k': ['https://drive.google.com/file/d/1sd53lVQg5txMwURR4u2YBGYEg4EbKM8a/view?usp=share_link', 'zip'],
    'RTG_2abrupt': ['https://drive.google.com/file/d/1YJRwEI04lLF7LQqeEx52o1Q3lE6p85xr/view?usp=share_link', 'zip'],
    'sensor': ['https://drive.google.com/file/d/11AZ92fWT8vDRoX6TOaAQgzcisHKbR9FW/view?usp=share_link', 'zip'],
    'moa': ['https://drive.google.com/file/d/1W4CR_QmrdGb8qFqmhP9UXSTh3k5azIv9/view?usp=share_link', 'jar'],
}

for name, file_info in z_files.items():
    out_file_name = name + '.' + file_info[1]
    url = file_info[0]
    print(f'Downloading {out_file_name} from : {url}')
    gdown.download(url, out_file_name, quiet=False, fuzzy=True)
    print(f'downloaded file name: {out_file_name}')
    if file_info[1].__eq__('zip'):
        with zipfile.ZipFile(out_file_name, 'r') as archive:
            for archive_file in archive.namelist():
                if archive_file.find('/' + name + '.csv') or archive_file.find('/' + name + '.arff'):
                    print(f'unzip: {archive_file} to ./data/')
                    archive.extract(archive_file, './data/')
        print(f'remove downloaded zip file: {out_file_name}')
        os.remove(out_file_name)
    elif file_info[1].__eq__('jar'):
        os.rename(out_file_name, 'jar/' + out_file_name)

shutil.rmtree("./data/__MACOSX")
