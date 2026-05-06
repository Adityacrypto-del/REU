Add-Type -AssemblyName System.IO.Compression.FileSystem

$pptxPath = "C:\Users\91948\Downloads\PPT Template.pptx"
$outPath = "C:\Users\91948\OneDrive\Attachments\ReU\template_extracted.txt"
$zip = [System.IO.Compression.ZipFile]::OpenRead($pptxPath)
$slides = $zip.Entries | Where-Object { $_.FullName -match 'ppt/slides/slide\d+\.xml$' } | Sort-Object { [int]($_.FullName -replace '.*slide(\d+)\.xml','$1') }

$output = ""
foreach ($slide in $slides) {
    $stream = $slide.Open()
    $reader = New-Object System.IO.StreamReader($stream)
    $xml = [xml]$reader.ReadToEnd()
    $reader.Close()
    $stream.Close()
    $texts = $xml.SelectNodes('//*[local-name()="t"]')
    $slideNum = $slide.FullName -replace '.*slide(\d+)\.xml','$1'
    $output += "=== SLIDE $slideNum ===`r`n"
    foreach ($t in $texts) {
        $output += $t.InnerText + "`r`n"
    }
    $output += "`r`n"
}
$zip.Dispose()

$output | Out-File -FilePath $outPath -Encoding utf8
Write-Host "Done! Output saved to $outPath"
