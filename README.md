# General Viewpoint Principle

We are investigating the Generic Viewpoint Principle (GVP), which formalizes the intuition that the human visual system perceives interpretations which are stable under small viewpoint perturbations.

GVP connects perception to Bayesian Occam’s Razor via the notion that interpretations requiring accidental conditions are penalized by the generic-view term in the scene probability equation, which favors explanations that account for the image over a broad range of generic variables (e.g., view, pose, light). Developing a rendering model that instantiates GVP scenarios offers a testable link between probabilistic generative models and human judgments. Specifically we ask the following question: Can a Bayesian generative model that marginalizes over generic viewpoints quantitatively predict human judgments of 2D vs 3D interpretations across classic and novel GVP stimuli?


We built a model in Julia that uses the GVP scene probability equation from Freeman to predict the dimensionality of rendered wireframes. These wireframes model canonical pairs (e.g. square and cube) with azimuth and elevation as the generic variables. We compare the model predictions to human predictions, which reveals information about human priors.

---

## File Layout and Module Descriptions

### `gvp_model.jl`
- Model that uses Chamfer distance for scoring and the GVP scene probability equation with generic variables azimuth and elevation.
- **Output**:  
  - `P(2D | image)`
  - `P(3D | image)`


### `infer_gen.jl`
- Integrates GLMakie with Gen. Uses a `wireframe_camera_model` with MH updates and `ImageGaussian` scoring to predict the best azimuth and elevation given an observed image.
- **Output**:  
  - `obs_img.png`
  - `vis.gif`


### `infer_drift_ransac.jl`
- Integrates GLMakie with Gen. Uses an RANSAC proposed intial prediction and Gaussian Drift updates to predict the best azimuth and elevation given an observed image.
- **Output**:  
  - `obs_img.png`
  - `vis.gif`


### `render.jl`
- Renders 40 images of randomly rendered 2D and 3D wireframes.

### `wireframe.jl`
- Prints the wireframes used in `render.jl`.

