# PINN-Arrh

Disclaimer: The code has been shared with the company's approval (also it's very unorganized since it's a proof of concept code) so here is an intro to the method : 

## A Physics-informed neural networks approach to modelling Li-Ion calendar ageing

This notebook contains some of the work I have done during my internship at **Serma Tech**, to **model the Calendar ageing** (ageing that takes place in storage) of Lithium-Ion  battery cells.

The models mostly seen in litterature/industry-standard are based on the Arrhenius equation that is generally used to model the speed of a reaction w.t.r to Temprature.

In our case, it looks something like this:

$$\frac{\partial C}{\partial t} = \frac{\alpha(T)}{ \sqrt{t}}$$

With:

$C$ : The cell capacity at instant $t$

$T$ : Temperature

$\alpha$ : An acceleration coefficient

$$\alpha(T) = a \exp\left(\frac{b}{T}\right)$$

  
Where $a$ and $b$ are two parameters to estimate (they change depending on other factors like the difference in cell type, manufacturing, initial State Of Charge and other parameters ...)


Our approach was to leverage the same equation plus testing data to train a standard neural network (MLP), using a custom loss function that we defined as such :


$$ L_{training} = a_1 L_{MSE} + a_2 L_{DE}+a_3 L_{Boundary}+a_4 L_{CstPenalty}$$


1- We derived the equation that has the solution above: 

$$\frac{\partial^2 C_{loss}}{\partial t}+\frac{1}{2t}\frac{\partial C_{loss}}{\partial t }=0$$

The loss function derived from this equation: 

$$ L_{DE} = \frac{1}{n}\sum_{i=1}^{n} |\frac{\partial^2 f}{\partial t}(T_i,t_i,SOC_i;\theta)+\frac{1}{2t_i}\frac{\partial f}{\partial t }(T_i,t_i,SOC_i;\theta)|$$

Where: f (T, t, SOC; θ) is the model's response to the input (T, t, SOC) and θ the model parameters.


2- We also penalize the constant solution to the previous equation, so that the model doesn't have a constant output which would minimize the $L_{DE}$ loss : 

$$ L_{CstPenalty} = \frac{1}{n}\sum|\frac{1}{\frac{\partial f}{\partial t}(T_i,t_i,SOC_i;\theta)}|$$

3- MSE loss to learn from training samples : 
 $$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(f(T_i,t_i,SOC_i;\theta) - C_i)^2$$

4- A loss to translate the boudary condition,  "no ageing at t=0"
$$L_{Boundary} = \sum|f(T_{synth},0,SOC_{synth};\theta)|$$

The coefficients $a_1$ , $a_2$ , $a_3$ , $a_4$ were optimized to balance the contribution of each loss and also the fact that the losses have different scales.

#### Results
- This approach improved the mean absolute error on a batch test cells, from 1,05%(industry standard method) to 0.38%(our method), (in % because we compute the capacity normalized with with starting value (in %)), in the context of the project it's was big difference.
- This project was in a consulting context, the client was more relieved that output of the model resembeled the standard method of the industry.
