Calibrator monitoring polynomials
=================================
Compute time and frequency polynomials from the VLA calibrator monitoring
program.  This module is to be run with Python v3.


Getting started
---------------
First, clone or download this repository and run

.. code-block:: bash

   pip install --user --requirement requirements.txt

Then add the module directory to your ``PYTHONPATH``.

To generate the plots, call:

.. code-block:: python

   from tcal_poly import (core, plotting)
   f_df = core.aggregate_flux_files()
   w_df = core.read_weather()
   plotting.plot_all_light_curves(f_df, bands=core.BANDS)
   plotting.plot_all_seds_rel(f_df)
   plotting.plot_all_weather_light_curves(
        f_df, w_df, fields=plotting.FSCALE_FIELDS, bands=core.bands,
   )


License
-------
The pipeline is authored by Brian Svoboda. The code and documentation is
released under the MIT License. A copy of the license is supplied in the
LICENSE file.


