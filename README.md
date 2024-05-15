# On the use of convolutional deep learning to predict shoreline change

Here you can find the alongshore-averaged cross-shore position time series for Tairua Beach, wave data, and code of the CNNs and CNN-LSTM models to reproduce the results in the paper by [Gomez-de la Pena et al. 2023](https://doi.org/10.5194/esurf-11-1145-2023). Authors: Eduardo Gomez-de la Pena, Giovanni Coco, Colin Whittaker, and Jennifer Montano. The original shoreline and wave data were curated by co-author JM and used in the paper by [Montaño et al. 2021](https://doi.org/10.1029/2020GL090587) . Earlier versions of the shoreline data set can be found at the [Coast and Ocean Collective website](https://coastalhub.science/data) .

To ensure that you have the appropriate requirements to run the Python scripts, please create a Mamba environment using the provided .yaml file ("cnns_4_schange.yaml"). To run the models, run scripts in /1run_models. To generate plots, run scripts in /2gen_plots. 


## Citation

```
@article{Gomez-delaPena2023,
  title={On the use of Convolutional Deep Learning to predict shoreline change},
  author={Gomez-de la Pena, Eduardo and Coco, Giovanni and Whittaker, Colin and Montano, Jennifer},
  journal={EGUsphere},
  volume={2023},
  pages={1--24},
  year={2023},
  DOI = {10.5194/egusphere-2023-958},
  publisher={Copernicus Publications G{\"o}ttingen, Germany}
}
