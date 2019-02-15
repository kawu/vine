{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}


module MWE 
  ( Sent(..)
  , train
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
import qualified Net.MWE2 as MWE
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



-- | Encoded dataset element
--
-- TODO: Move to `Net.ArcGraph` module.
--
data Elem d = Elem
  { graph :: Net.Graph (R d)
    -- ^ Input graph
  , labMap :: M.Map Net.Arc (R 2)
    -- ^ Target labels
--   , tokMap :: M.Map Cupt.TokID G.Vertex
--     -- ^ Token ID -> graph vertices mapping
  } deriving (Show, Generic, Binary)


-- | Convert the given Cupt file to a training dataset element.
--   * `d` -- embedding size (dimension)
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> (Elem d, M.Map Cupt.TokID G.Vertex)
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
  -> Elem d
createElem nodes arcs = Elem
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
             then MWE.mwe
             else MWE.notMwe
        )


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
  -> IO (SGD.DataSet (Elem d))
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


-- data TrainConfig = TrainConfig
--   { 
--   } deriving (Show, Eq, Ord)


-- -- | Train the MWE identification network.
-- train
--   :: (Cupt.MweTyp -> Bool)
--     -- ^ MWE type (category) selection method
--   -> [Sent 300]
--   -> IO (Net.Param 300 2)
-- train mweSel cupt = do
--   net0 <- Net.new 300 2
--   Net.trainProg (gdCfg trainData) 1 net0
--   where
--     trainData = do
--       sent <- cupt
--       let (Elem{..}, _) = mkElem mweSel sent
--       return (graph, labMap)
--     -- Gradient descent configuration
--     gdCfg dataSet depth = Mom.Config
--       { iterNum = 200
--       , gradient = pure . BP.gradBP (Net.netError dataSet depth)
--       , quality = pure . BP.evalBP (Net.netError dataSet depth)
--       , reportEvery = 10
--       , gain0 = 0.01 / fromIntegral (depth+1)
--       , tau = 100
--       , gamma = 0.0 ** fromIntegral (depth+1)
--       }


-- | Train the MWE identification network.
train
  :: (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> FilePath
    -- ^ Path to store temporary results
  -> [Sent 300]
    -- ^ Training dataset
  -> IO (Net.Param 300 2)
train mweSel tmpPath cupt = do
  dataSet <- mkDataSet mweSel tmpPath cupt
  net0 <- Net.new 300 2
  trainProgSGD sgdCfg dataSet 1 net0
  where
    sgdCfg depth = SGD.Config
      { iterNum = 30
      , batchSize = 1
      , gradient = \xs net -> BP.gradBP (Net.netError (map unElem xs) depth) net
      , quality = \x net -> BP.evalBP (Net.netError [unElem x] depth) net
      , reportEvery = 1
      , gain0 = 0.01 -- / fromIntegral (depth+1)
      , tau = 5
      , gamma = 0.9 -- ** fromIntegral (depth+1)
      }
    unElem Elem{..} = (graph, labMap)


-- | Progressive training
trainProgSGD
  :: (KnownNat d, KnownNat c)
  => (Int -> SGD.Config (Net.Param d c) (Elem d))
    -- ^ Gradient descent config, depending on the chosen depth
  -> SGD.DataSet (Elem d)
    -- ^ Training dataset
  -> Int
    -- ^ Maximum depth
  -> Net.Param d c
    -- ^ Initial params
  -> IO (Net.Param d c)
trainProgSGD gdCfg dataSet maxDepth =
  go 0
  where
    go depth net
      | depth > maxDepth =
          return net
      | otherwise = do
          putStrLn $ "# depth = " ++ show depth
          net' <- SGD.sgd net dataSet (gdCfg depth)
          go (depth+1) net'


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tag many sentences
tagMany
  :: Net.Param 300 2  -- ^ Network parameters
  -> Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> [Sent 300]       -- ^ Cupt sentences
  -> IO ()
tagMany net mweTyp cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tag net mweTyp sent


-- | Tag (output the result on stdin).
tag
  :: Net.Param 300 2  -- ^ Network parameters
  -> Cupt.MweTyp      -- ^ MWE type (category) to tag with
  -> Sent 300         -- ^ Cupt sentence
  -> IO ()
tag net mweTyp sent = do

  forM_ (M.toList arcMap) $ \(e, v) -> do
    when (isMWE v) (printArc e)

  where

    (elem, tokMap) = mkElem (const False) sent
    arcMap = Net.eval net 1 (graph elem)
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
