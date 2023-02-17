# bci_adv_defense
This project evaluated some typical adversarial defense approaches in BCIs.

## Evaluated defenses

Following defenses were evaluated:
- Natural training: ``` NT.py ```
- Adversarial training: ``` AT.py ```
- TRADES: ``` TRADES.py ```
- SCR: ```SRC.py```
- HYDRA: ``` HYDRA.py ```
- Stochastic activation pruning: ``` stochastic_activation_pruning.py ```
- Input transformation: ``` input_transform.py ```
- Random self ensemble: ``` random_self_ensemble.py ```
- Self ensemble adversarial training: ``` self_ensemble_AT.py ```

## Attacks
Two white-box attacks and two black-box attacks with $\ell_{\infty}$ and $\ell_2$ norm were used to evaluate defenses, which can be found in the file ``` attack_lib.py ```. 

## Evaluation
The file ``` evaluation.py ``` can be used for evaluation after model trained with defense. For example, the evaluation of EEGNet trained with AT against $\ell_{\infty}$ untargeted attacks in within-subject setup is as follows:  
``` python3 evaluation.py --model EEGNet --defense AT --distance inf --target False ```
