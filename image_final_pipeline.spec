# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['image_final_pipeline.py'],
             pathex=['/path to/image_final_pipeline'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)


dict_modelfiles = Tree('/path to/image_final_pipeline/model_files')
a.datas += dict_modelfiles

dict_utils = Tree('/path to/image_final_pipeline/utils')
a.datas += dict_utils


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='image_final_pipeline',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
