; installer.iss — Instalador do "Análise de Marcha" (Inno Setup 6)
;
; PRÉ-REQUISITO: rodar antes o build do painel:  pyinstaller app_marcha.spec
;   (isso gera a pasta dist\AnaliseMarcha que este instalador empacota)
;
; COMO GERAR O setup.exe:
;   1. Instale o Inno Setup (grátis): https://jrsoftware.org/isinfo.php
;   2. Abra este arquivo (installer.iss) no Inno Setup e clique em "Build" (ou tecla F9),
;      ou pela linha de comando:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
;   3. O instalador sai em:  Output\AnaliseMarcha_Setup.exe   <-- ESTE é o arquivo a enviar.

#define AppName "Analise de Marcha"
#define AppVersion "1.0.0"
#define AppPublisher "Edilson Borba"
#define AppExe "AnaliseMarcha.exe"

[Setup]
AppId={{B7E4B6B0-1A2C-4E3D-9F10-ANALISEMARCHA01}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\AnaliseMarcha
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Instala só em Windows 64-bit
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=AnaliseMarcha_Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupIconFile=icone.ico
UninstallDisplayIcon={app}\icone.ico
; Precisa de admin para instalar em Arquivos de Programas
PrivilegesRequired=admin

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na Área de Trabalho"; GroupDescription: "Atalhos:"

[Files]
; Empacota TODA a pasta gerada pelo PyInstaller (exe + _internal).
Source: "dist\AnaliseMarcha\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Ícone do app na pasta de instalação (usado nos atalhos e na desinstalação).
Source: "icone.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\icone.ico"
Name: "{group}\Desinstalar {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; IconFilename: "{app}\icone.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Abrir {#AppName} agora"; Flags: nowait postinstall skipifsilent

[Messages]
brazilianportuguese.WelcomeLabel2=Isto instalará o [name] no seu computador.%n%nNa primeira vez que abrir, o programa pode baixar componentes de GPU e os modelos (precisa de internet uma vez). Depois funciona offline.
