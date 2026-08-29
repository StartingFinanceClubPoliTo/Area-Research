param(
    [Parameter(Mandatory = $true)][string]$PagesDirectory,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [int]$Columns = 3,
    [int]$Rows = 4
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null

$pages = Get-ChildItem -LiteralPath $PagesDirectory -Filter '*.jpg' | Sort-Object {
    if ($_.BaseName -match '(\d+)$') { [int]$Matches[1] } else { [int]::MaxValue }
}

$thumbWidth = 260
$thumbHeight = 368
$labelHeight = 24
$gutter = 16
$sheetWidth = $Columns * ($thumbWidth + $gutter) + $gutter
$sheetHeight = $Rows * ($thumbHeight + $labelHeight + $gutter) + $gutter
$perSheet = $Columns * $Rows
$font = New-Object System.Drawing.Font('Arial', 11, [System.Drawing.FontStyle]::Bold)
$labelBrush = [System.Drawing.Brushes]::Black
$background = [System.Drawing.Color]::FromArgb(235, 235, 235)

try {
    for ($offset = 0; $offset -lt $pages.Count; $offset += $perSheet) {
        $sheetIndex = [int]($offset / $perSheet) + 1
        $sheet = New-Object System.Drawing.Bitmap($sheetWidth, $sheetHeight)
        $graphics = [System.Drawing.Graphics]::FromImage($sheet)
        try {
            $graphics.Clear($background)
            $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $batchEnd = [Math]::Min($offset + $perSheet, $pages.Count)
            for ($i = $offset; $i -lt $batchEnd; $i++) {
                $slot = $i - $offset
                $column = $slot % $Columns
                $row = [int]($slot / $Columns)
                $x = $gutter + $column * ($thumbWidth + $gutter)
                $y = $gutter + $row * ($thumbHeight + $labelHeight + $gutter)
                $pageImage = [System.Drawing.Image]::FromFile($pages[$i].FullName)
                try {
                    $graphics.DrawImage($pageImage, $x, $y, $thumbWidth, $thumbHeight)
                } finally {
                    $pageImage.Dispose()
                }
                $pageNumber = if ($pages[$i].BaseName -match '(\d+)$') { $Matches[1] } else { $pages[$i].BaseName }
                $graphics.DrawString("PDF page $pageNumber", $font, $labelBrush, $x, $y + $thumbHeight + 2)
            }
            $outputPath = Join-Path $OutputDirectory ('contact-{0:D2}.png' -f $sheetIndex)
            $sheet.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
        } finally {
            $graphics.Dispose()
            $sheet.Dispose()
        }
    }
} finally {
    $font.Dispose()
}

Write-Output "Created $([Math]::Ceiling($pages.Count / $perSheet)) contact sheets for $($pages.Count) pages."
