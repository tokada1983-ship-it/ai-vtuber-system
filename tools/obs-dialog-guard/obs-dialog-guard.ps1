# obs-dialog-guard.ps1
# OBS Studio crash-confirmation dialog auto-dismiss helper.
# Detects the "OBS Studio crash detected" QMessageBox of obs64.exe via
# UIAutomation and presses ONLY the "Normal Mode" button (never Safe Mode).
# Press strategy: InvokePattern first, verify the dialog is gone, then fall
# back to a mouse click at the button element's real bounding rectangle.
# Must run in the interactive user session (UIAutomation cannot see the
# desktop from session 0). Logs to <YOUR_PATH>\logs\obs-dialog-guard.log.
param(
  [int]$TimeoutSec = 120,
  [int]$PollMs = 500,
  [string]$LogFile = "<YOUR_PATH>\logs\obs-dialog-guard.log"
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'

function Write-Log([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Write-Output $line
  try {
    $dir = Split-Path $LogFile -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
  } catch { }
}

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class GuardMouse {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
}
"@

function Test-Port4455 {
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect('127.0.0.1', 4455, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(800, $false)
    if ($ok) { $c.EndConnect($iar) }
    $c.Close()
    return $ok
  } catch { return $false }
}

function Find-CrashDialog {
  # Returns @{ Window=...; NormalButton=... } if the OBS crash dialog is up,
  # otherwise $null. Safety: requires BOTH a Safe-Mode button and a
  # Normal-Mode button inside the same QMessageBox owned by obs64.
  $procs = Get-Process obs64 -ErrorAction SilentlyContinue
  if (-not $procs) { return $null }
  foreach ($proc in $procs) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
      [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc.Id)
    $wins = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
      [System.Windows.Automation.TreeScope]::Children, $cond)
    foreach ($w in $wins) {
      if ($w.Current.ClassName -ne 'QMessageBox') { continue }
      $btnCond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button)
      $btns = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants, $btnCond)
      $safeBtn = $null; $normalBtn = $null
      foreach ($b in $btns) {
        $n = $b.Current.Name
        if ($n -match 'セーフ' -or $n -match 'Safe') { $safeBtn = $b }
        if ($n -match '通常' -or $n -match 'Normal') { $normalBtn = $b }
      }
      if ($safeBtn -and $normalBtn) {
        return @{ Window = $w; NormalButton = $normalBtn }
      }
    }
  }
  return $null
}

function Test-DialogGone($dlg) {
  Start-Sleep -Milliseconds 1500
  try {
    $name = $dlg.Window.Current.Name
    # If the element is gone, accessing it throws or the search finds nothing.
    $again = Find-CrashDialog
    return ($null -eq $again)
  } catch { return $true }
}

function Press-NormalButton($dlg) {
  $btn = $dlg.NormalButton
  Write-Log ("dialog detected: title='{0}' normal-button='{1}'" -f $dlg.Window.Current.Name, $btn.Current.Name)
  # 1) InvokePattern
  try {
    $pat = $null
    if ($btn.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$pat)) {
      Write-Log 'attempting InvokePattern on Normal-Mode button'
      $pat.Invoke()
      Write-Log 'InvokePattern returned'
    } else {
      Write-Log 'InvokePattern not available on button'
    }
  } catch {
    Write-Log ("InvokePattern failed: " + $_.Exception.Message)
  }
  if (Test-DialogGone $dlg) { Write-Log 'dialog dismissed by InvokePattern'; return $true }
  # 2) Fallback: click at the element's real bounding rectangle.
  try {
    $fresh = Find-CrashDialog
    if ($fresh) { $btn = $fresh.NormalButton }
    $r = $btn.Current.BoundingRectangle
    if ($r.Width -gt 0 -and $r.Height -gt 0) {
      $x = [int]($r.X + $r.Width / 2); $y = [int]($r.Y + $r.Height / 2)
      [GuardMouse]::SetCursorPos($x, $y) | Out-Null
      Start-Sleep -Milliseconds 250
      [GuardMouse]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
      [GuardMouse]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
      Write-Log "clicked Normal-Mode button at element rect ($x,$y)"
    } else {
      Write-Log 'button has no valid bounding rectangle; cannot click'
    }
  } catch {
    Write-Log ("click fallback failed: " + $_.Exception.Message)
  }
  if (Test-DialogGone $dlg) { Write-Log 'dialog dismissed by click'; return $true }
  Write-Log 'dialog still present after both press methods'
  return $false
}

$result = 'error'
try {
  Write-Log "guard start (timeout=${TimeoutSec}s)"
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  $dismissed = $false
  $attempted = $false
  while ((Get-Date) -lt $deadline) {
    $dlg = Find-CrashDialog
    if ($dlg) {
      $attempted = $true
      if (Press-NormalButton $dlg) { $dismissed = $true; break }
      Start-Sleep -Seconds 2
      continue
    }
    if (-not $attempted -and (Test-Port4455)) {
      Write-Log 'no dialog and port 4455 already responding; nothing to do'
      $result = 'not-needed'
      break
    }
    if ($attempted -and -not $dismissed) {
      # dialog vanished between polls without a confirmed press
      $dismissed = $true
      break
    }
    Start-Sleep -Milliseconds $PollMs
  }
  if ($dismissed) {
    $wsDeadline = (Get-Date).AddSeconds(60)
    $ws = $false
    while ((Get-Date) -lt $wsDeadline) {
      if (Test-Port4455) { $ws = $true; break }
      Start-Sleep -Seconds 1
    }
    Write-Log "dialog dismissed; websocket4455=$ws"
    $result = 'dismissed'
  } elseif ($result -eq 'error') {
    $result = 'timeout'
  }
} catch {
  Write-Log ("guard exception: " + $_.Exception.Message)
  $result = 'error'
}
Write-Log "guard done result=$result"
if ($result -eq 'dismissed' -or $result -eq 'not-needed') { exit 0 } else { exit 1 }
