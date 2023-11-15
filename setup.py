from setuptools import setup, find_packages

setup(
    name = 'Aletheia',
    description = 'Image steganalysis using state-of-the-art machine learning techniques',
    license = 'MIT',
    author = 'Daniel Lerch Hostalot',
    author_email = 'dlerch@gmail.com',
    download_url = 'https://github.com/daniellerch/aletheia',
    #packages = ['aletheialib', 'aletheialib.options', 'models'],
    packages=find_packages(),
    version = '0.2',
    scripts = ['aletheia.py'],
    install_requires = ['imageio', 'numpy', 'scipy', 'tensorflow', 'scikit-learn',
                        'pandas', 'hdf5storage', 'h5py', 'matplotlib',
                        'steganogan', 'python-magic', 'efficientnet', 'Pillow'],
    include_package_data = True
)

