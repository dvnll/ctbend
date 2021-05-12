A python package to model pointing corrections for Cherenkov telescopes.

The package is divided into two modules, ctbendbase and ctbendtrainer. 
The base module, ctbendbase, depends only on a very basic set of standard python modules. The training module, ctbendtrainer, has extended dependencies but 
is only necessary for the model building and data analysis. The division into two modules allows for a flexible training module, to be set up in a computing environment for data analysis,
while the application of bending models is possible with minimal dependencies in computing environments for science data taking.

CTbend allows for flexible definitions of pointing models, which must all derive from ctbendbase.CTBendBase. Some predefined examples for models are given in ctbendbase.CTBend.

Pointing run data for the creation of pointing models with ctbendtrainer are expected to be in the format defined by ctbendbase.PointingData.PointingDataset. 
A PointingDataset is a collection of PointingData, each of which holds one pointing datum. 
A pointing datum consists of the ctbendbase.PointingData.CCDCoordinate of the star and the telescope as well as the ctbendbase.PointingData.DriveCoordinate of the telescope drive system.

A pointing model is loaded with the factory method bending_factory:

```
import ctbendbase

model_json = from file, database etc.
pointing_model = ctbendbase.bending_factory(model_json)
```

Once loaded, the pointing corrections for the telescope pointing towards (az, el) in degrees are obtained via:

```
delta_azimuth = pointing_model.delta_azimuth(az, el)
delta_elevation = pointing_model.delta_elevation(az, el)

corrected_azimuth = az + delta_azimuth
corrected_elevation = el + delta_elevation
```

Now, point the telescope to (azimuth, elevation)=(corrected_azimuth, corrected_elevation), in degrees.

For citations, please refer to the (upcoming) proceedings to the ICRC2021:

```
@article{ctbend,
    author = "G. Spengler et al.",
    title = "{CTbend: A Bayesian open-source framework to model pointing corrections for Cherenkov telescopes}
    journal = "PoS",
    volume = "ICRC2021",
    year = "2021"
}
```
