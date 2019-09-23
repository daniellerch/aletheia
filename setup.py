from setuptools import setup

setup(
    name='Aletheia',
    description='Image steganalysis using state-of-the-art machine learning techniques',
    license='MIT',
    author='Daniel Lerch Hostalot',
    author_email='dlerch@gmail.com',
    download_url='https://github.com/daniellerch/aletheia',
    packages=['aletheialib'],
    version='0.1',
    scripts=['aletheia.py'],
    install_requires=['numpy', 'scipy', 'tensorflow', 'keras', 'scikit-learn',
                      'scikit-image', 'pandas', 'hdf5storage', 'h5py']

)

