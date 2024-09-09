
********************************************************************************
* Save CSV output as dta & append across model types
********************************************************************************
* -- 1. Metrics for full model
import delimited using "en_metrics_all.csv", clear
drop if model == ""
save "en_metrics_all.dta",replace

import delimited using "gbm_metrics_all.csv", clear
drop if model == ""
save "gbm_metrics_all.dta",replace

import delimited using "glm_metrics_all.csv", clear
drop if model == ""
save "glm_metrics_all.dta",replace

import delimited using "multiglm_metrics_all.csv", clear
drop if model == ""
save "multiglm_metrics_all.dta",replace

import delimited using "rf_metrics_all.csv", clear
drop if model == ""
save "rf_metrics_all.dta",replace

import delimited using "baseline_metrics.csv", clear
drop if model == ""
save "baseline_metrics.dta",replace

use "en_metrics_all.dta", clear
append using "gbm_metrics_all.dta" ///
"glm_metrics_all.dta" ///
"multiglm_metrics_all.dta" ///
"rf_metrics_all.dta" ///
"baseline_metrics.dta"


save "metrics_all.dta", replace

* -- 2. Metrics for clinician model
import delimited using "en_metrics_c.csv", clear
drop if model == ""
save "en_metrics_c.dta",replace

import delimited using "gbm_metrics_c.csv", clear
drop if model == ""
save "gbm_metrics_c.dta",replace

import delimited using "glm_metrics_c.csv", clear
drop if model == ""
save "glm_metrics_c.dta",replace

import delimited using "multiglm_metrics_c.csv", clear
drop if model == ""
save "multiglm_metrics_c.dta",replace

import delimited using "rf_metrics_c.csv", clear
drop if model == ""
save "rf_metrics_c.dta",replace

use "en_metrics_c.dta", clear
append using "gbm_metrics_c.dta" ///
"glm_metrics_c.dta" ///
"multiglm_metrics_c.dta" ///
"rf_metrics_c.dta"


save "metrics_c.dta", replace

********************************************************************************
* Derive summary statistics
********************************************************************************
use "metrics_all.dta", clear

* -- A. Full model 
* 1. Mean performance metrics in test set
foreach model in MultiGLM ElasticNet RF GBM SingleGLM {
display "Mean and 95%CI for `model' performance"
ci mean auc if model=="`model'"
ci mean recall if model=="`model'" 
ci mean specificity if model=="`model'"
ci mean ppv if model=="`model'"
ci mean npv if model=="`model'"
ci mean brierscore if model=="`model'"
}

* 2. Calculation for Offender status at baseline
ci mean auc if model=="Baseline"
ci mean recall if model=="Baseline" 
ci mean specificity if model=="Baseline"
ci mean ppv if model=="Baseline"
ci mean npv if model=="Baseline"
ci mean brierscore if model=="Baseline"

* 3. Calculation of SHAP values 
	* -- 3-1. Generate composite variables
egen test_shap_site = rowtotal(test_shap_site_greenwich test_shap_site_hackney test_shap_site_leeds test_shap_site_merton test_shap_site_peterborough test_shap_site_reading test_shap_site_sheffield test_shap_site_trafford)

egen test_shap_referrer = rowtotal(test_shap_referrer_education test_shap_referrer_other test_shap_referrer_policetriage test_shap_referrer_socialcare test_shap_referrer_yos)

egen test_shap_ses = rowtotal(test_shap_ses_low test_shap_ses_medium)

	* -- 3-2. Calculate shap values in the TEST SET
foreach model in MultiGLM ElasticNet RF GBM {
putexcel set shap_values_test.xlsx, sheet(`model') modify
putexcel A1=("Model") B1=("Variable") C1=("Mean") D1=("lb") E1=("ub")
local row=2
	foreach var in test_shap_age test_shap_iq test_shap_numchild test_shap_p_ghq_totalt1 test_shap_p_conn_learlangtscoret test_shap_p_conn_adhdtscoret1 test_shap_p_icu_totalt1 test_shap_p_sdq_totalimpactt1 test_shap_p_sdq_emott1 test_shap_p_sdq_cdt1 test_shap_p_sdq_hypert1 test_shap_p_sdq_peerrelt1 test_shap_p_sdq_prosoct1 test_shap_cts25totalt1 test_shap_p_alab_posparentt1 test_shap_p_alab_parinvt1 test_shap_p_alab_mont1 test_shap_p_alab_corpunt1 test_shap_p_alab_incdist1 test_shap_p_face_cohesiondimensi test_shap_p_face_flexibilitydime test_shap_p_face_fcommt1 test_shap_p_face_fsatt1 test_shap_p_loeb_totalt1 test_shap_yp_smf_totalt1 test_shap_yp_icu_totalt1 test_shap_yp_srd_del_exsib_vart1 test_shap_yp_srd_del_exsib_volt1 test_shap_yp_srd_submis_vart1 test_shap_yp_srd_submis_volt1 test_shap_yp_srd_peerillsubt1 test_shap_yp_srd_peerdelt1 test_shap_yp_lee_totalt1 test_shap_yp_abas_totalt1 test_shap_yp_sdq_totalimpactt1 test_shap_yp_sdq_emott1 test_shap_yp_sdq_cdt1 test_shap_yp_sdq_hypert1 test_shap_yp_sdq_peerrelt1 test_shap_yp_sdq_prosoct1 test_shap_yp_youthmatscalet1 test_shap_unauthorisedabst1 test_shap_noexclt1 test_shap_totaldaysexclt1 test_shap_alloffs_pre test_shap_off_noff_offender         test_shap_referrer_education test_shap_time_mst test_shap_gender_male test_shap_onsetcd_late test_shap_married_singleorwidowe test_shap_pqual_yes test_shap_pemploy_10 test_shap_soffend_yes test_shap_poffend_yes test_shap_yeduc_main_yes test_shap_yeduc_spec_yes test_shap_ysen_yes test_shap_ethnicity_20 test_shap_yaccom_20 test_shap_dawba_dep_10 test_shap_dawba_adhd_10 test_shap_dawba_cdt_10 test_shap_site test_shap_referrer test_shap_ses{

ci mean `var' if model=="`model'"
putexcel A`row'=("`model'")
putexcel B`row'=("`var'")
putexcel C`row'=(r(mean))
putexcel D`row'=(r(lb))
putexcel E`row'=(r(ub))

local ++row
}
}

* -- B. Clinician model 
use "metrics_c.dta", clear
* 1. Mean performance metrics in test set
foreach model in MultiGLM ElasticNet RF GBM SingleGLM {
display "Mean and 95%CI for `model' performance"
ci mean auc if model=="`model'"
ci mean recall if model=="`model'" 
ci mean specificity if model=="`model'"
ci mean ppv if model=="`model'"
ci mean npv if model=="`model'"
ci mean brierscore if model=="`model'"
}

* 2. Calculation of SHAP values 
	* -- 2-1. Generate composite variables
egen test_shap_site = rowtotal(test_shap_site_greenwich test_shap_site_hackney test_shap_site_leeds test_shap_site_merton test_shap_site_peterborough test_shap_site_reading test_shap_site_sheffield test_shap_site_trafford)

egen test_shap_referrer = rowtotal(test_shap_referrer_education test_shap_referrer_other test_shap_referrer_policetriage test_shap_referrer_socialcare test_shap_referrer_yos)

egen test_shap_ses = rowtotal(test_shap_ses_low test_shap_ses_medium)

	* -- 2-2. Calculate shap values in the TEST SET
foreach model in MultiGLM ElasticNet RF GBM {
putexcel set shap_values_test.xlsx, sheet(`model'_c) modify
putexcel A1=("Model") B1=("Variable") C1=("Mean") D1=("lb") E1=("ub")
local row=2
	foreach var in test_shap_age test_shap_iq test_shap_numchild test_shap_alloffs_pre test_shap_site_greenwich test_shap_site_hackney test_shap_site_leeds test_shap_site_merton test_shap_site_peterborough test_shap_site_reading test_shap_site_sheffield test_shap_site_trafford test_shap_referrer_education test_shap_referrer_other test_shap_referrer_policetriage test_shap_referrer_socialcare test_shap_referrer_yos test_shap_soffend_yes test_shap_poffend_yes test_shap_gender_male test_shap_ethnicity_20 test_shap_married_singleorwidowe test_shap_pqual_yes test_shap_pemploy_10 test_shap_ses_low test_shap_ses_medium test_shap_yaccom_20 test_shap_time_mst test_shap_dawba_dep_10 test_shap_dawba_adhd_10 test_shap_dawba_cdt_10 test_shap_off_noff_offender test_shap_site test_shap_referrer test_shap_ses {

ci mean `var' if model=="`model'"
putexcel A`row'=("`model'")
putexcel B`row'=("`var'")
putexcel C`row'=(r(mean))
putexcel D`row'=(r(lb))
putexcel E`row'=(r(ub))

local ++row
}
}