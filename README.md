# Segmentation_Archaeological_Structures_Nefud
## Semi-automated detection of Holocene archaeological structures along the southern edge of the Nefud Desert
by Hatton et al.

(short description)

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Prerequisites

## References

(reference items)

## Further information

### Running code locally
### Running code on HPC system that uses SLURM

You need to first create an Apptainer container (with the required packages)

'''
apptainer build --fakeroot nvidia_pytorch.sif nvidia_pytorch.def
'''

Where .def is the definition file and .sif is the container image that is created. 

You will need to upload the code and data to the HPC and then run this code by calling the slurm script as follows:

'''
sbatch run_job.sh
'''


(Free text no longer than 10000 characters)

## Dataset persistent identifier

## Contact
Amy Hatton - hatton@gea.mpg.de






