import Router from 'express'
import multer from 'multer'
import { auth } from '../middleware/auth'
import path from 'path'

export const router = Router()

const storage = multer.diskStorage({
  destination(req, file, cb) {
    cb(null, 'uploads/')
  },
  filename(req, file, cb) {
    const ext = path.extname(file.originalname)
    cb(null, `${Date.now()}-${Math.round(Math.random() * 1E9)}${ext}`)
  },
})

const upload = multer({
  storage,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB
  },
})

router.post('/upload-image', auth, upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      res.send({ status: 'Fail', message: '没有文件被上传', data: null })
      return
    }
    const data = {
      fileKey: req.file.filename,
    }
    res.send({ status: 'Success', message: '文件上传成功', data })
  }
  catch (error) {
    res.send(error)
  }
})
