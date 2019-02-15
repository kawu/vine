{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}


module MWE 
  ( Sent(..)

  -- * Training
  , TrainConfig(..)
  , defTrainCfg
  , train

  -- * Tagging
  , tag
  , tagMany
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)

import           Control.Monad (guard, when, forM_)

import qualified System.Directory as D
import           System.FilePath ((</>))

import           Numeric.LinearAlgebra.Static.Backprop (R)
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.Backprop as BP

import           Dhall -- (Interpret)
import qualified Data.Aeson as JSON

import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as B
-- import           Codec.Compression.Zlib (compress, decompress)
import qualified Data.IORef as R

import qualified Format.Cupt as Cupt
import qualified Net.ArcGraph as Net
import           Net.ArcGraph (Elem)
-- import qualified Net.MWE2 as MWE
import qualified Embedding as Emb
import qualified GradientDescent.Momentum as Mom
import qualified SGD as SGD

-- import Debug.Trace (trace)


----------------------------------------------
-- Data
----------------------------------------------


-- | Input sentence
data Sent d = Sent
  { cuptSent :: Cupt.Sent
    -- ^ The .cupt sentence
  , wordEmbs :: [R d]
    -- ^ The corresponding word embeddings
  } deriving (Show, Generic, Binary)



-- | Convert the given Cupt file to a training dataset element.
--   * `d` -- embedding size (dimension)
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> (Elem d 2, M.Map Cupt.TokID G.Vertex)
mkElem mweSel Sent{..} =

  (createElem nodes arcs, idMap)

  where

    -- Helpers
    idMap = M.fromList
      [ (Cupt.tokID tok, i)
      | tok <- cuptSent
      , let (Cupt.TokID i) = Cupt.tokID tok
      ]
    tokID tok = idMap M.! Cupt.tokID tok
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSent 
      ]
    tokPar tok = tokMap M.! Cupt.dephead tok
    isRoot tok = Cupt.dephead tok == Cupt.TokID 0
    getMweIDs tok
      = S.fromList
      . map fst
      . filter (mweSel . snd)
      $ Cupt.mwe tok

    -- Graph nodes and labeled arcs
    nodes = do
      (tok, vec) <- zip cuptSent wordEmbs
      return (tokID tok, vec)
    arcs = do
      tok <- cuptSent
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
          -- The arc label is `True` if the token and its parent are annoted as
          -- a part of the same MWE of the appropriate MWE category
          arcLabel =
            if (not . S.null)
                 ((getMweIDs tok) `S.intersection` (getMweIDs par))
               then True
               else False
      return ((tokID tok, tokID par), arcLabel)


-- | Create a dataset element based on nodes and labeled arcs.
createElem
  :: (KnownNat d)
  => [(G.Vertex, R d)]
  -> [(Net.Arc, Bool)]
  -> Elem d 2
createElem nodes arcs = Net.Elem
  { graph = graph
  , labMap = valMap
  }
  where
    vertices = [v | (v, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      (map fst arcs)
    lbMap = M.fromList nodes
    graph = Net.Graph
      { Net.graphStr = gStr
      , Net.graphInv = G.transposeG gStr
      , Net.labelMap = lbMap }
    valMap = M.fromList $ do
      (arc, isMwe) <- arcs
      return 
        ( arc
        , if isMwe
             then mwe
             else notMwe
        )


-- | Is MWE or not?
mwe, notMwe :: R 2
notMwe = LA.vector [1, 0]
mwe = LA.vector [0, 1]


----------------------------------------------
-- Dataset
----------------------------------------------


-- | Create a dataset from a list of sentences in a given directory.
-- TODO: Make it error-safe/aware
mkDataSet
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> FilePath
    -- ^ Path to store temporary results
  -> [Sent d]
  -> IO (SGD.DataSet (Elem d 2))
mkDataSet mweSel path xs = do
  D.doesDirectoryExist path >>= \case
    False -> return ()
    True -> error "Directory already exists!"
  D.createDirectory path

  nRef <- R.newIORef 0
  forM_ (zip [0..] xs) $ \(ix, sent) -> do
    R.modifyIORef' nRef (+1)
    Bin.encodeFile (path </> show ix) (mkElem mweSel sent)

  n <- R.readIORef nRef
  return $ SGD.DataSet
    { size = n
    , elemAt = \ix -> Bin.decodeFile (path </> show ix)
    }


----------------------------------------------
-- Training
----------------------------------------------


-- globalDepth :: Int 
-- globalDepth = 0 


data TrainConfig = TrainConfig
  { trainDepth :: Integer
    -- ^ Graph net recursion depth
  , trainIterNum :: Integer
    -- ^ Number of iterations (for SGD)
  , trainBatchSize :: Integer
    -- ^ Batch size (for SGD)
  , trainReportEvery :: Double
    -- ^ For SGD
  , trainGain0 :: Double
    -- ^ For SGD
  , trainTau :: Double
    -- ^ For SGD
  , trainGamma :: Double
    -- ^ For SGD
  } deriving (Show, Eq, Ord, Generic, Interpret)


-- instance Interpret TrainConfig

instance JSON.FromJSON TrainConfig
instance JSON.ToJSON TrainConfig where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions


-- | Default training configuration
defTrainCfg :: TrainConfig
defTrainCfg = TrainConfig
  { trainDepth = 0
  , trainIterNum = 10
  , trainBatchSize = 1
  , trainReportEvery = 1
  , trainGain0 = 0.01
  , trainTau = 5
  , trainGamma = 0.9
  }


-- | Train the MWE identification network.
train
  :: TrainConfig
    -- ^ General training confiration
  -> FilePath
    -- ^ Directory for temporary storage
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent 300]
    -- ^ Training dataset
  -> Net.Param 300 2
    -- ^ Initial network
  -> IO (Net.Param 300 2)
train cfg tmpDir mweTyp cupt net0 = do
  dataSet <- mkDataSet (== mweTyp) tmpDir cupt
  -- net0 <- Net.new 300 2
  -- trainProgSGD sgdCfg dataSet globalDepth net0
  SGD.sgd net0 dataSet
    (sgdCfg . fromIntegral $ trainDepth cfg)
  where
    sgdCfg depth = SGD.Config
      { iterNum = fromIntegral $ trainIterNum cfg
      , batchSize = fromIntegral $ trainBatchSize cfg
      , gradient = \xs -> BP.gradBP (Net.netError xs depth)
      , quality = \x -> BP.evalBP (Net.netError [x] depth)
      , reportEvery = trainReportEvery cfg
      , gain0 = trainGain0 cfg -- / fromIntegral (depth+1)
      , tau = trainTau cfg
      , gamma = trainGamma cfg -- ** fromIntegral (depth+1)
      }


-- -- | Progressive training
-- trainProgSGD
--   :: (KnownNat d, KnownNat c)
--   => (Int -> SGD.Config (Net.Param d c) (Elem d c))
--     -- ^ Gradient descent config, depending on the chosen depth
--   -> SGD.DataSet (Elem d c)
--     -- ^ Training dataset
--   -> Int
--     -- ^ Maximum depth
--   -> Net.Param d c
--     -- ^ Initial params
--   -> IO (Net.Param d c)
-- trainProgSGD gdCfg dataSet maxDepth =
--   go 0
--   where
--     go depth net
--       | depth > maxDepth =
--           return net
--       | otherwise = do
--           putStrLn $ "# depth = " ++ show depth
--           net' <- SGD.sgd net dataSet (gdCfg depth)
--           go (depth+1) net'


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tag many sentences
tagMany
  :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Net.Param 300 2  -- ^ Network parameters
  -> Int              -- ^ Depth (see also `trainDepth`)
  -> [Sent 300]       -- ^ Cupt sentences
  -> IO ()
tagMany mweTyp net depth cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tag mweTyp net depth sent


-- | Tag (output the result on stdin).
tag
  :: Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Net.Param 300 2  -- ^ Network parameters
  -> Int              -- ^ Depth (see also `trainDepth`)
  -> Sent 300         -- ^ Cupt sentence
  -> IO ()
tag mweTyp net depth sent = do

  forM_ (M.toList arcMap) $ \(e, v) -> do
    when (isMWE v) (printArc e)

  where

    (elem, tokMap) = mkElem (const False) sent
    arcMap = Net.eval net depth (Net.graph elem)
    isMWE statVect =
      let vect = LA.unwrap statVect
          val = vect `LAD.atIndex` 1
       in val > 0.5

    wordMap = M.fromList $ do
      tok <- cuptSent sent
      return (tokMap M.! Cupt.tokID tok, Cupt.orth tok)

    printArc (p, q) = do
      T.putStr (wordMap M.! p)
      T.putStr " => "
      T.putStrLn (wordMap M.! q)
