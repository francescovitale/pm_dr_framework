del /F /Q Input\PP\EventData\*
del /F /Q Input\PF\NormativeModel\*
for /D %%p IN ("Input\PF\EventLogs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
)
del /F /Q Input\DA\Data\*	
del /F /Q Input\AE\Data\*

for /D %%p IN ("Output\PP\EventLogs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
)
del /F /Q Output\PF\Data\*
del /F /Q Output\DA\Metrics\*
del /F /Q Output\DA\Model\*
del /F /Q Output\AE\*
del /F /Q Results\*
for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)