:: Options:
:: n_reps=<integer>
:: datasets=[PDC2020,PDC2021,COVAS,ERTMS]
:: n_traces_per_log=<integer>
:: test_percentage=[0,1]
:: validation_percentage=[0,1]
:: pm_fe_techniques=[alignment_based_cc,token_based_cc,directly_follows,n_grams]
:: dr_techniques=[PCA,KPCA,AE]
:: nb_pm_fe_techniques=[alignment_based_cc,token_based_cc]

set n_reps=1
set datasets=PDC2020 PDC2021 ERTMS
set n_traces_per_log=5
set test_percentage=0.25
set validation_percentage=0.2

:: Parameters of non-baseline techniques
set pm_fe_techniques=alignment_based_cc token_based_cc directly_follows n_grams
set dr_techniques=PCA KPCA AE


:: Parameters of baseline techniques
set nb_pm_fe_techniques=alignment_based_cc token_based_cc

call clean_environment

for %%s in (%datasets%) do (
	
	mkdir Results\%%s
	
	del /F /Q Input\PP\EventData\*
	del /F /Q Input\PF\NormativeModel\*
	
	copy Data\%%s\PetriNet\* Input\PF\NormativeModel
	copy Data\%%s\EventData\Output\* Input\PP\EventData
	
	@echo Non-baseline techniques
	
	for %%f in (%pm_fe_techniques%) do (
		for %%d in (%dr_techniques%) do (
		
			mkdir Results\%%s\%%f\%%d
			
			for /l %%x in (1, 1, %n_reps%) do (
				for /D %%p IN ("Input\PF\EventLogs\*") DO (
					del /s /f /q %%p\*.*
					for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				)
				del /F /Q Input\DA\Data\*
				
				for /D %%p IN ("Output\PP\EventLogs\*") DO (
					del /s /f /q %%p\*.*
					for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				)
				del /F /Q Output\PF\Data\*
				del /F /Q Output\DA\Metrics\*
				
				python preprocessing.py %n_traces_per_log% %validation_percentage% %test_percentage%
				
				copy Output\PP\EventLogs\Training\* Input\PF\EventLogs\Training
				copy Output\PP\EventLogs\Test\* Input\PF\EventLogs\Test
				copy Output\PP\EventLogs\Validation\* Input\PF\EventLogs\Validation
				
				python pm_fe.py %%f
				
				copy Output\PF\Data\* Input\DA\Data
				
				python dr_ad.py %%d
				
				copy Output\DA\Metrics\Metrics.txt Results\%%s\%%f\%%d
				ren Results\%%s\%%f\%%d\Metrics.txt Metrics_%%x.txt
				
			)
		)
	)
	
	@echo Baseline techniques
	
	for %%f in (%nb_pm_fe_techniques%) do (
	
		mkdir Results\%%s\%%f\NO_DR
		
		for /l %%x in (1, 1, %n_reps%) do (
			for /D %%p IN ("Input\PF\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
			)
			del /F /Q Input\DA\Data\*
			
			for /D %%p IN ("Output\PP\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
			)
			del /F /Q Output\PF\Data\*
			
			python preprocessing.py %n_traces_per_log% %validation_percentage% %test_percentage%
				
			copy Output\PP\EventLogs\Training\* Input\PF\EventLogs\Training
			copy Output\PP\EventLogs\Test\* Input\PF\EventLogs\Test
			copy Output\PP\EventLogs\Validation\* Input\PF\EventLogs\Validation
				
			python pm_fe.py %%f
				
			copy Output\PF\Data\* Input\DA\Data
				
			python dr_ad.py NO_DR
			
			copy Output\DA\Metrics\Metrics.txt Results\%%s\%%f\NO_DR
			ren Results\%%s\%%f\NO_DR\Metrics.txt Metrics_%%x.txt
			
		)
	)
	

)
