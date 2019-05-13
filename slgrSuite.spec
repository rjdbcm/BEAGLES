# -*- mode: python -*-

block_cipher = None


a = Analysis(['slgrSuite.py'],
             binaries=[],
             datas=[('data/predefined_classes.txt', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

a.datas += Tree('./data', prefix='data')
a.datas += Tree('./backend', prefix='backend')

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='slgrSuite',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='SLGR-Suite')
app = BUNDLE(coll,
             name='SLGR-Suite.app',
             icon='resources/icons/app.icns',
             bundle_identifier=None)
