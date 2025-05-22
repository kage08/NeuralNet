from setuptools import setup, find_packages

setup(
    name='neural_network_suite',
    version='0.1.0',
    author='AI Agent', 
    author_email='agent@example.com',
    description='A suite of neural network implementations with vectorized operations.',
    long_description=open('README.md').read(), # Assumes README.md
    long_description_content_type='text/markdown', # If README is markdown
    url='https://github.com/placeholder/neural-network-suite', # Placeholder URL
    packages=find_packages(where="src"), # This anticipates moving code to a 'src' directory
    package_dir={"": "src"}, # This anticipates moving code to a 'src' directory
    install_requires=[
        'numpy>=1.18.0',
        'scikit-learn>=0.22.0',
        'matplotlib>=3.0.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Assuming MIT License, adjust if different
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
