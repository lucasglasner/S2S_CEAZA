## <center> S2S ocean forecast analysis
---

This repository contains some postprocess routines for a intraseasonal kelvin-wave forecast study. The repository is structured on different jupyter notebooks, each with some particular analysis or purpose. Every notebook filename has a numerical tag in relation with the order of each notebook to work. First of all there is some data creation and the postprocess. Naturally the postprocess wont work if the previous datasets arent created. 

The project consist of the statistical evaluation of ocean forecast skills. In particular oceanic teleconections between the equator and east Pacific coastal band. With that objetive in mind the procedure goes as follow: 

1. Create equatorial and coastal masks for all the different grids involved
2. Create equatorial and coastal hovmollers of the global data
3. Compute climatologies and geophysical indexes of interest (i.e kelvin wave activity, MJO, etc)
4. Postprocess and formal analysis (skill evaluation)