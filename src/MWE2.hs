{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}

{-# OPTIONS_GHC -fno-warn-orphans #-}


-- | The top-level module of the VMWE identification tool.  Provides plumbing
-- between various more specialized modules:
--
--   * `N.Graph` -- ANN over graphs
--   * `Format.Cupt` -- .cupt format support
--   * `Numeric.SGD` -- stochastic gradient descent (external library)
--   * ...


module MWE2
  ( Sent(..)

  -- * New
  , N.new
  , N.newO
  , N.Typ

  -- * Training
  , Config(..)
  , Method(..)
  , trainO
  , trainT
  , depRelsIn
  , posTagsIn

  -- * Tagging
  , TagConfig(..)
  , tag
  , tagT
  , tagMany
  , tagManyO
  , tagManyT
  ) where


import           Prelude hiding (elem)

import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
-- import qualified GHC.TypeNats as Nats

import           Control.Monad (guard, forM_)
import           Control.DeepSeq (NFData)

import           Lens.Micro ((^.))

-- import qualified System.Directory as D
-- import           System.FilePath ((</>))

import           Numeric.LinearAlgebra.Static.Backprop (R)
-- import qualified Numeric.LinearAlgebra.Static as LA
-- import qualified Numeric.LinearAlgebra as LAD
import           Numeric.Backprop (Reifies, W, BVar, (^^.))
import qualified Numeric.Backprop as BP

import           Dhall -- (Interpret)
import           Dhall.Core (Expr(..))
import qualified Dhall.Map as Map
-- import qualified Data.Aeson as JSON

import qualified Data.Proxy as Proxy
import qualified Data.Foldable as Fold
import           Data.Semigroup (Max(..))
import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Ord as Ord
-- import           Data.Function (on)
import qualified Data.List as List
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy.IO as L
import           Data.Binary (Binary)
-- import qualified Data.Binary as Bin
-- import qualified Data.ByteString.Lazy as B
-- import           Codec.Compression.Zlib (compress, decompress)
-- import qualified Data.IORef as R

import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Type as SGD
import qualified Numeric.SGD.ParamSet as SGD
import qualified Numeric.SGD.DataSet as SGD
import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Adam as Adam

import qualified Format.Cupt as Cupt
import qualified Graph
import qualified Net.Graph2 as N
import qualified Net.Graph2.BiComp as B
import qualified Net.Graph2.UniComp as U
import qualified Net.Input as I

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


-- -- | Convert the given .cupt file to a training dataset element.  The token IDs
-- -- are used as vertex identifiers in the resulting graph.
-- --
-- --   * @d@ -- embedding size (dimension)
-- --
-- mkElem
--   :: (KnownNat d)
--   => (Cupt.MweTyp -> Bool)
--     -- ^ MWE type (category) selection method
--   -> Sent d
--     -- ^ Input sentence
--   -> N.Elem d DepRel POS
-- mkElem mweSel sent =
-- 
--   createElem nodes arcs
-- 
--   where
-- 
--     -- Cupt sentence with ID range tokens removed
--     cuptSentF = filter
--       (\tok -> case Cupt.tokID tok of
--                  Cupt.TokID _ -> True
--                  _ -> False
--       )
--       (cuptSent sent)
-- 
--     -- A map from token IDs to tokens
--     tokMap = M.fromList
--       [ (Cupt.tokID tok, tok)
--       | tok <- cuptSentF 
--       ]
--     -- The parent of the given token
--     tokPar tok = tokMap M.! Cupt.dephead tok
--     -- Is the token the root?
--     isRoot tok = Cupt.dephead tok == Cupt.TokID 0
--     -- The IDs of the MWEs present in the given token
--     getMweIDs tok
--       = S.fromList
--       . map fst
--       . filter (mweSel . snd)
--       $ Cupt.mwe tok
-- 
--     -- Graph nodes: a list of token IDs and the corresponding vector embeddings
--     nodes = do
--       (tok, vec) <- zip cuptSentF (wordEmbs sent)
--       let node = Graph.Node
--             { nodeEmb = vec
--             , nodeLab = Cupt.upos tok
--             -- | TODO: could be lemma?
--             , nodeLex = Cupt.orth tok
--             }
--       return (tokID tok, node)
--     -- Labeled arcs of the graph
--     arcs = do
--       tok <- cuptSentF
--       -- Check the token is not a root
--       guard . not $ isRoot tok
--       let par = tokPar tok
--           -- The arc value is `True` if the token and its parent are annoted as
--           -- a part of the same MWE of the appropriate MWE category
--           isMwe = (not . S.null)
--             (getMweIDs tok `S.intersection` getMweIDs par)
--       return ((tokID tok, tokID par), Cupt.deprel tok, isMwe)
-- 
-- 
-- -- | Create a dataset element based on nodes and labeled arcs.
-- --
-- -- Works under the assumption that incoming/outgoing arcs are ,,naturally''
-- -- ordered in accordance with their corresponding vertex IDs (e.g., for two
-- -- children nodes x, v, the node with lower ID precedes the other node).
-- --
-- createElem
--   :: (KnownNat d)
--   => [(G.Vertex, Graph.Node d POS)]
--   -> [(Graph.Arc, DepRel, Bool)]
--   -> N.Elem d DepRel POS
-- createElem nodes arcs0 = N.Elem
--   { graph = graph
--   , nodMap = _nodMap
--   , arcMap = _arcMap
--   }
--   where
--     arcs = List.sortBy (Ord.comparing _1) arcs0
--     vertices = [v | (v, _) <- nodes]
--     gStr = G.buildG
--       (minimum vertices, maximum vertices)
--       (map _1 arcs)
--     graph = verify . Graph.mkAsc $ Graph.Graph
--       { Graph.graphStr = gStr
--       , Graph.graphInv = G.transposeG gStr
--       , Graph.nodeLabelMap = M.fromList $ nodes
--           -- map (\(x, e, pos) -> (x, Graph.Node e pos)) nodes
--       , Graph.arcLabelMap = M.fromList $
--           map (\(x, y, _) -> (x, y)) arcs
--       }
--     _arcMap = M.fromList $ do
--       (arc, _arcLab, isMwe) <- arcs
--       return (arc, if isMwe then 1.0 else 0.0)
--     _nodMap = M.fromListWith max . concat $ do
--       ((v, w), _arcLab, isMwe) <- arcs
--       let mwe = if isMwe then 1.0 else 0.0
--       return [(v, mwe), (w, mwe)]
--     _1 (x, _, _) = x
--     verify g
--       | Graph.isAsc g = g
--       | otherwise = error "MWE.createElem: constructed graph not ascending!"


-- | Convert the given .cupt file to a training dataset element.  The token IDs
-- are used as vertex identifiers in the resulting graph.
--
--   * @d@ -- embedding size (dimension)
--
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> N.Elem (R d)
mkElem mweSel sent0 =

  List.foldl' markMWE
    (createElem nodes arcs)
    ( filter (mweSel . Cupt.mweTyp')
    . M.elems
    . Cupt.retrieveMWEs
    -- TODO: what would happen if a MWE were marked on a token with ID range!?
    $ cuptSent sent
    )

  where

    -- Sentence with ID range tokens removed
    sent = discardMerged sent0

--     WARNING: The code below was incorrect!  We cannot remove tokens but
--     leave the embeddings!
--     -- Cupt sentence with ID range tokens removed
--     cuptSentF = filter
--       (\tok -> case Cupt.tokID tok of
--                  Cupt.TokID _ -> True
--                  _ -> False
--       )
--       (cuptSent sent)

    -- A map from token IDs to tokens
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSent sent
      ]
    -- The parent of the given token
    tokPar tok = tokMap M.! Cupt.dephead tok
    -- Is the token the root?
    isRoot tok = Cupt.dephead tok == Cupt.TokID 0
    -- The IDs of the MWEs present in the given token
    getMweIDs tok
      = S.fromList
      . map fst
      . filter (mweSel . snd)
      $ Cupt.mwe tok

    -- Graph nodes: a list of token IDs and the corresponding vector embeddings
    nodes = do
      (tok, vec) <- zipSafe (cuptSent sent) (wordEmbs sent)
      let node = (tok, vec)
--       let node = Graph.Node
--             { nodeEmb = vec
--             , nodeLab = Cupt.upos tok
--             -- | TODO: could be lemma?
--             , nodeLex = Cupt.orth tok
--             }
          isMwe = (not . S.null) (getMweIDs tok)
      return (tokID tok, node, isMwe)
    -- Labeled arcs of the graph
    arcs = do
      tok <- cuptSent sent
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
          -- TODO: there should be no need to mark arcs as MWE in this version
          -- of `mkElem`, since `markMWEs` is executed afterwards anyway.
          --
          -- The arc value is `True` if the token and its parent are annoted as
          -- a part of the same MWE of the appropriate MWE category.
          isMwe = (not . S.null)
            (getMweIDs tok `S.intersection` getMweIDs par)
      -- return ((tokID tok, tokID par), Cupt.deprel tok, isMwe)
      return ((tokID tok, tokID par), isMwe)


discardMerged :: (KnownNat d) => Sent d -> Sent d
discardMerged sent0 
  = uncurry Sent . unzip
  $ filter cond 
  $ zip (cuptSent sent0) (wordEmbs sent0)
  where
    cond (tok, emb) =
      case Cupt.tokID tok of
        Cupt.TokID _ -> True
        _ -> False


-- | Mark MWE in the given dataset element.
markMWE :: N.Elem d -> Cupt.Mwe -> N.Elem d
markMWE el mwe =
  List.foldl' markArc el (S.toList arcSet)
  where
    markArc el arc = el {N.arcMap = M.insert arc 1.0 (N.arcMap el)}
    arcSet =
      N.treeConnectAll
        (N.graph el)
        (S.fromList . map tokID . S.toList $ Cupt.mweToks mwe)


-- | Create a dataset element based on nodes and labeled arcs.
--
-- Works under the assumption that incoming/outgoing arcs are ,,naturally''
-- ordered in accordance with their corresponding vertex IDs (e.g., for two
-- children nodes x, v, the node with lower ID precedes the other node).
--
-- Version of `createElem` in which node labels are explicitly handled.
--
createElem
  :: (KnownNat d)
  => [(G.Vertex, (Cupt.Token, R d), Bool)]
  -> [(Graph.Arc, Bool)]
  -> N.Elem (R d)
createElem nodes arcs0 = N.Elem
  { graph = graph
  , nodMap = _nodMap
  , arcMap = _arcMap
  , tokMap = M.fromList [(v, tok) | (v, (tok, _), _) <- nodes]
  }
  where
    -- arcs = List.sortBy (Ord.comparing _1) arcs0
    arcs = List.sortBy (Ord.comparing fst) arcs0
    vertices = [v | (v, _, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      (map fst arcs)
    graph = verify . Graph.mkAsc $ Graph.Graph
      { Graph.graphStr = gStr
      , Graph.graphInv = G.transposeG gStr
      , Graph.nodeLabelMap = M.fromList $
          map (\(x, y, _) -> (x, snd y)) nodes
      , Graph.arcLabelMap = M.fromList $
          map (\(x, _) -> (x, ())) arcs
      }
    _arcMap = M.fromList $ do
      -- (arc, _arcLab, isMwe) <- arcs
      (arc, isMwe) <- arcs
      return (arc, if isMwe then 1.0 else 0.0)
    _nodMap = M.fromList $ do
      (v, _, isMwe) <- nodes
      return (v, if isMwe then 1.0 else 0.0)
    _1 (x, _, _) = x
    verify g
      | Graph.isAsc g = g
      | otherwise = error "MWE.createElem: constructed graph not ascending!"




-- -- | Create an unlabeled dataset element based on nodes and arcs.
-- --
-- -- Works under the assumption that incoming/outgoing arcs are ,,naturally''
-- -- ordered in accordance with their corresponding vertex IDs (e.g., for two
-- -- children nodes x, v, the node with lower ID precedes the other node).
-- createElem'
--   :: (KnownNat d)
--   => [(G.Vertex, Graph.Node d POS)]
--   -> [(Graph.Arc, DepRel)]
--   -> N.Elem d DepRel POS
-- createElem' nodes arcs0 = N.Elem
--   { graph = graph
--   , nodMap = _nodMap
--   , arcMap = _arcMap
--   }
--   where
--     arcs = List.sortBy (Ord.comparing fst) arcs0
--     vertices = [v | (v, _) <- nodes]
--     gStr = G.buildG
--       (minimum vertices, maximum vertices)
--       (map fst arcs)
--     graph = verify . Graph.mkAsc $ Graph.Graph
--       { Graph.graphStr = gStr
--       , Graph.graphInv = G.transposeG gStr
--       , Graph.nodeLabelMap = M.fromList nodes
--       , Graph.arcLabelMap = M.fromList arcs
--       }
--     _arcMap = M.fromList $ do
--       (arc, _arcLab) <- arcs
--       return (arc, 0.0)
--     _nodMap = M.fromList $ do
--       (v, _nodeLab) <- nodes
--       return (v, 0.0)
--     verify g
--       | Graph.isAsc g = g
--       | otherwise = error "MWE.createElem: constructed graph not ascending!"


-- -- | Is MWE or not?
-- mwe, notMwe :: R 2
-- notMwe = LA.vector [1, 0]
-- mwe = LA.vector [0, 1]


-- | Token ID
tokID :: Cupt.Token -> Int
tokID tok =
  case Cupt.tokID tok of
    Cupt.TokID i -> i
    Cupt.TokIDRange _ _ ->
      error "MWE.tokID: token ID ranges not supported"


----------------------------------------------
-- Training
--
-- TODO: mostly copy from MWE
----------------------------------------------


-- Some orphan `Interpret` instances
instance Interpret Mom.Config
instance Interpret Ada.Config
instance Interpret Adam.Config
instance Interpret SGD.Config
-- instance Interpret SGD.Method


data Method
  = Momentum Mom.Config
  | AdaDelta Ada.Config
  | Adam Adam.Config
  deriving (Generic)

instance Interpret Method where
  autoWith _ = rmUnion_1 genericAuto


data Config = Config
  { sgd :: SGD.Config
  , method :: Method
  , probTyp :: N.ProbTyp
  } deriving (Generic)

instance Interpret Config


-- | Select the sgd method
toSGD
  :: (SGD.ParamSet p)
  => Config
    -- ^ Configuration
  -> Int
    -- ^ Dataset size
  -> (e -> p -> p)
    -- ^ Gradient
  -> SGD.SGD IO e p
toSGD Config{..} dataSize grad =
  case method of
    Momentum cfg -> Mom.momentum (Mom.scaleTau iterNumEpoch cfg) grad
    AdaDelta cfg -> Ada.adaDelta cfg grad
    Adam cfg -> Adam.adam (Adam.scaleTau iterNumEpoch cfg) grad
  where
    iterNumEpoch = SGD.iterNumPerEpoch sgd dataSize


-- | Depdency relation
type DepRel = T.Text


-- | POS tag
type POS = T.Text


-- | Train the MWE identification network.
train
  :: ( KnownNat d
     , B.BiComp d comp
     , SGD.ParamSet comp, NFData comp
     )
  => Config
    -- ^ General training confiration
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent d]
    -- ^ Training dataset
  -> comp
--   -> N.Param 300
    -- ^ Initial network
  -> IO comp
--   -> IO (N.Param 300)
train cfg mweTyp cupt net0 = do
  -- dataSet <- mkDataSet (== mweTyp) tmpDir cupt
  let cupt' = map (mkElem (== mweTyp)) cupt
  net' <- SGD.withDisk cupt' $ \dataSet -> do
    putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
    -- net0 <- N.new 300 2
    -- trainProgSGD sgd dataSet globalDepth net0
    SGD.runIO (sgd cfg)
      (toSGD cfg (SGD.size dataSet) (SGD.batchGradPar gradient))
      (SGD.reportObjective quality dataSet)
      dataSet net0
--   N.printParam net'
  return net'
  where
    gradient x = BP.gradBP (N.netError (probTyp cfg) [fmap BP.auto x])
    quality x = BP.evalBP (N.netError (probTyp cfg) [fmap BP.auto x])


-- | Extract dependeny relations present in the given dataset.
depRelsIn :: [Cupt.GenSent mwe] -> S.Set DepRel
depRelsIn =
  S.fromList . concatMap extract 
  where
    extract = map Cupt.deprel


-- | Extract dependeny relations present in the given dataset.
posTagsIn :: [Cupt.GenSent mwe] -> S.Set POS
posTagsIn =
  S.fromList . concatMap extract 
  where
    extract = map Cupt.upos


----------------------------------------------
-- Working with opaque network architecture
--
-- TODO: copy from MWE
----------------------------------------------


-- | Train the opaque MWE identification network.
trainO
  :: Config
    -- ^ General training confiration
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent 300]
    -- ^ Training dataset
  -> N.Opaque 300 DepRel POS
    -- ^ Initial network
  -> IO (N.Opaque 300 DepRel POS)
trainO cfg mweTyp cupt net =
  case net of
    N.Opaque t p -> N.Opaque t <$> train cfg mweTyp cupt p


-- | Tag sentences with the opaque network.
tagManyO
  :: TagConfig
  -> N.Opaque 300 DepRel POS
  -> [Sent 300]
  -> IO ()
tagManyO cfg = \case
  N.Opaque _ p -> tagMany cfg p


----------------------------------------------
-- Trainin with input transformations
----------------------------------------------


-- -- | Train the MWE identification network.
-- trainI
--   :: forall inp comp d i.
--      ( KnownNat d, KnownNat i
--      , I.Input inp d i
--      , SGD.ParamSet inp, NFData inp
--      , B.BiComp i comp
--      , SGD.ParamSet comp, NFData comp
--      )
--   => Config
--     -- ^ General training confiration
--   -> Proxy.Proxy i
--     -- ^ A proxy to learn the internal dimension
--     -- TODO: could be a part of the config, I guess?
--   -> Cupt.MweTyp
--     -- ^ Selected MWE type (category)
--   -> [Sent d]
--     -- ^ Training dataset
--   -> inp -> comp
--     -- ^ Initial networks
--   -> IO (inp, comp)
-- --   -> IO (N.Param 300)
-- trainI cfg _proxy mweTyp cupt inp0 net0 = do
--   -- dataSet <- mkDataSet (== mweTyp) tmpDir cupt
--   let cupt' = map (mkElem (== mweTyp)) cupt
--   SGD.withDisk cupt' $ \dataSet -> do
--     putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
--     SGD.runIO (sgd cfg)
--       (toSGD cfg (SGD.size dataSet) (SGD.batchGradPar gradient))
--       quality dataSet (inp0, net0)
--   where
--     gradient x (inp0, net0) = BP.gradBP2 (netError x) inp0 net0
--     quality x (inp, net) = BP.evalBP2 (netError x) inp0 net0
--     netError 
--       :: forall s. (Reifies s W)
--       => N.Elem (R d)
--       -> BVar s inp
--       -> BVar s comp
--       -> BVar s Double
--     netError x inp net =
--       let toksEmbs = N.tokens x
--           embs' = I.runInput inp toksEmbs :: [BVar s (R i)]
--           x' = N.replace embs' x
--        in N.netError (probTyp cfg) [x'] net


-- | Train the MWE identification network.
trainT
  :: Config
    -- ^ General training confiration
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent 300]
    -- ^ Training dataset
  -> N.Transparent
    -- ^ Initial networks
  -> (N.Transparent -> IO ())
    -- ^ Action to execute at the end of each epoch
  -> IO N.Transparent
--   -> IO (N.Param 300)
trainT cfg mweTyp cupt tra0 action = do
  -- dataSet <- mkDataSet (== mweTyp) tmpDir cupt
  let cupt' = map (mkElem (== mweTyp)) cupt
  SGD.withDisk cupt' $ \dataSet -> do
    putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
    SGD.runIO (sgd cfg)
      (toSGD cfg (SGD.size dataSet) (SGD.batchGradPar gradient))
      (reportAndExec dataSet) dataSet tra0
  where 
    gradient x = BP.gradBP (N.netErrorT (probTyp cfg) x)
    quality x = BP.evalBP (N.netErrorT (probTyp cfg) x)
    reportAndExec dataSet tra = do
      x <- SGD.reportObjective quality dataSet tra
      action tra
      return x


----------------------------------------------
-- Tagging
--
-- TODO: copy from MWE, apart from `tag`
----------------------------------------------


-- | Tagging configuration
data TagConfig = TagConfig
  { mweThreshold :: Double
    -- ^ The minimum probability to consider an arc a MWE component
    -- (with 0.5 being the default)
  , mweTyp :: Cupt.MweTyp
    -- ^ MWE category to annotate
  , mweGlobal :: Bool
    -- ^ Use global (tree) inference; if so, the `mweThreshold` option is
    -- ignored
  , mweConstrained :: Bool
    -- ^ Constrained global inference; implies `mweGlobal`
  } deriving (Show, Eq, Ord)


-- | Tag sentences based on the given configuration and multi/quad-affine
-- network.  The output is directed to stdout.
tagMany
  :: ( KnownNat d
     , B.BiComp d comp
     -- , SGD.ParamSet comp, NFData comp
     )
  => TagConfig
  -> comp
                      -- ^ Network parameters
  -> [Sent d]       -- ^ Cupt sentences
  -> IO ()
tagMany tagCfg net cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tag tagCfg net sent


-- | Tag a single sentence with the given network.
tag
  :: ( KnownNat d
     , B.BiComp d comp
     )
  => TagConfig
  -> comp             -- ^ Network parameters
  -> Sent d           -- ^ Cupt sentence
  -> IO ()
tag tagCfg net sent = do
  L.putStrLn $ Cupt.renderPar [Cupt.abstract sent']
  where
    elem = mkElem (const False) sent
    tagF
      | mweConstrained tagCfg =
          N.treeTagConstrained (N.graph elem)
      | mweGlobal tagCfg =
          N.treeTagGlobal (N.graph elem)
      | otherwise = N.tagGreedy mweChoice
    mweChoice ps = geoMean ps >= mweThreshold tagCfg
    labeling = tagF . N.evalRaw net $ N.graph elem
    sent' = annotate (mweTyp tagCfg) (cuptSent sent) labeling


-- -- | Tag a single sentence with the given network.
-- tag'
--   :: ( KnownNat d
--      , B.BiComp d comp
--      , U.UniComp d comp'
--      )
--   => TagConfig
--   -> comp             -- ^ Network parameters
--   -> comp'            -- ^ Network parameters (uni)
--   -> Sent d           -- ^ Cupt sentence
--   -> IO ()
-- tag' tagCfg net netU sent = do
--   L.putStrLn $ Cupt.renderPar [Cupt.abstract sent']
--   where
--     elem = mkElem (const False) sent
--     tagF
--       | mweConstrained tagCfg =
--           N.treeTagConstrained' (N.graph elem)
--       | mweGlobal tagCfg =
--           N.treeTagGlobal' (N.graph elem)
--       | otherwise =
--           error "tag': greedy not implemented"
--           -- N.tagGreedy mweChoice
--     mweChoice ps = geoMean ps >= mweThreshold tagCfg
--     labeling = tagF
--       (N.evalRaw net (N.graph elem))
--       (N.evalRawUni netU (N.graph elem))
--     sent' = annotate (mweTyp tagCfg) (cuptSent sent) labeling


-- -- !!! TODO MARKED !!!
-- -- | Tag a single sentence with the given network.
-- tagT
--   :: TagConfig
--   -> N.Transparent   -- ^ Network parameters
--   -> Sent 300        -- ^ Cupt sentence
--   -> IO ()
-- tagT cfg net sent =
--   tag' cfg (net ^. N.biaMod) (net ^. N.uniMod) $ Sent
--     { cuptSent = cuptSent sent
--     , wordEmbs
--         = I.evalTransform (net ^. N.traMod)
--         . I.evalInput (net ^. N.inpMod)
--         $ zipSafe (cuptSent sent) (wordEmbs sent)
--     }


zipSafe :: [a] -> [b] -> [(a, b)]
zipSafe xs ys
  | length xs == length ys = zip xs ys
  | otherwise = error "zipSafe: length not the same!"


-- | Tag a single sentence with the given network.
tagT
  :: TagConfig
  -> N.Transparent   -- ^ Network parameters
  -> Sent 300        -- ^ Cupt sentence
  -> IO ()
tagT tagCfg net sent =
  L.putStrLn $ Cupt.renderPar [Cupt.abstract sent']
  where
    elem = N.evalInp (mkElem (const False) sent) net
    tagF
      | mweConstrained tagCfg =
          N.treeTagConstrained' (N.graph elem)
      | mweGlobal tagCfg =
          N.treeTagGlobal' (N.graph elem)
      | otherwise =
          error "tag': greedy not implemented"
          -- N.tagGreedy mweChoice
    mweChoice ps = geoMean ps >= mweThreshold tagCfg
    labeling = tagF
      (N.evalRaw (net ^. N.biaMod) (N.graph elem))
      (N.evalRawUni (net ^. N.uniMod) (N.graph elem))
    sent' = annotate (mweTyp tagCfg) (cuptSent sent) labeling


-- -- !!! TODO MARKED !!!
-- -- | Tag a single sentence with the given network.
-- tagT
--   :: TagConfig
--   -> N.Transparent   -- ^ Network parameters
--   -> Sent 300        -- ^ Cupt sentence
--   -> IO ()
-- tagT cfg net sent =
--   tag cfg (net ^. N.biaMod) $ Sent
--     { cuptSent = cuptSent sent
--     , wordEmbs 
--         = I.evalTransform (net ^. N.traMod)
--         . I.evalInput (net ^. N.inpMod)
--         $ zip (cuptSent sent) (wordEmbs sent)
--     }


-- -- | Tag a single sentence with the given network.
-- tagT
--   :: TagConfig
--   -> N.Transparent   -- ^ Network parameters
--   -> Sent 300        -- ^ Cupt sentence
--   -> IO ()
-- tagT cfg net sent =
--   tag cfg (net ^. N.biaMod) sent


-- | Tag sentences with the opaque network.
tagManyT
  :: TagConfig
  -> N.Transparent
  -> [Sent 300]
  -> IO ()
tagManyT cfg net cupt = do
  forM_ cupt $ \sent -> do
    T.putStr "# "
    T.putStrLn . T.unwords $ map Cupt.orth (cuptSent sent)
    tagT cfg net sent


-- | Average of the list of numbers
average :: Floating a => [a] -> a
average xs = sum xs / fromIntegral (length xs)
{-# INLINE average #-}


-- | Geometric mean
geoMean :: Floating a => [a] -> a
geoMean xs =
  product xs ** (1.0 / fromIntegral (length xs))
  -- consider also the version in the log scale:
  -- exp . (/ fromIntegral (length xs)) . sum $ map log xs
{-# INLINE geoMean #-}


----------------------------------------------
-- Annotation with both arc and node labels
----------------------------------------------


-- | Annotate the sentence with the given MWE type, given the specified arc and
-- node labeling.
annotate
  :: Cupt.MweTyp
    -- ^ MWE type to annotate with
  -> Cupt.Sent
    -- ^ .cupt sentence to annotate
  -> N.Labeling Bool
    -- ^ Node/arc labeling
  -> Cupt.Sent
annotate mweTyp cupt N.Labeling{..} =

  map enrich cupt

  where

    -- Enrich the token with new MWE information
    enrich tok = Prelude.maybe tok id $ do
      Cupt.TokID i <- return (Cupt.tokID tok)
      mweId <- M.lookup i mweIdMap
      let newMwe = (mweId, mweTyp)
      return tok {Cupt.mwe = newMwe : Cupt.mwe tok}

    -- Determine the set of MWE nodes and arcs
    nodSet = trueKeys nodLab
    arcSet = trueKeys arcLab

    -- The set of keys with `True` values
    trueKeys m = S.fromList $ do
      (x, val) <- M.toList m
      guard val
      return x

    -- Determine the mapping from nodes to new MWE id's
    ccs = findConnectedComponents arcSet
    mweIdMap = M.fromList . concat $ do
      (cc, mweId) <- zip ccs [maxMweID cupt + 1 ..]
      (v, w) <- S.toList cc
      return $ filter ((`S.member` nodSet) . fst)
        [(v, mweId), (w, mweId)]


----------------------------------------------
-- Annotation continued
----------------------------------------------


-- | Given a set of graph arcs, determine all the connected arc subsets in the
-- corresponding graph.
findConnectedComponents
  :: S.Set Graph.Arc
  -> [S.Set Graph.Arc]
findConnectedComponents arcSet
  | S.null arcSet = []
  | otherwise
      -- Some components can be empty!  Perhaps because the graph is sparse?
      = filter (not . S.null)
      $ map (arcsInTree arcSet) (G.components graph)
  where
    vertices = S.toList (nodesIn arcSet)
    graph = G.buildG
      (minimum vertices, maximum vertices)
      (S.toList arcSet)


-- | Determine the set of arcs in the given connected graph component.
-- TODO: This could be done much more efficiently!
arcsInTree
  :: S.Set Graph.Arc
    -- ^ The set of all arcs
  -> G.Tree G.Vertex
    -- ^ Connected component
  -> S.Set Graph.Arc
arcsInTree arcSet cc = S.fromList $ do
  (v, w) <- S.toList arcSet
  guard $ v `S.member` vset
  guard $ w `S.member` vset
  return (v, w)
  where
    vset = S.fromList (Fold.toList cc)


-- | The set of nodes in the given arc set.
nodesIn :: S.Set Graph.Arc -> S.Set G.Vertex
nodesIn arcSet =
  (S.fromList . concat)
    [[v, w] | (v, w) <- S.toList arcSet]


-- | Determine the maximum mwe ID present in the given sentence.
maxMweID :: Cupt.Sent -> Int
maxMweID =
  getMax . Fold.foldMap mweID
  where
    mweID tok = case Cupt.mwe tok of
      [] -> Max 0
      xs -> Max . maximum $ map fst xs


----------------------------------------------
-- Dhall Utils
--
-- TODO: copy from MWE
----------------------------------------------


-- | Remove the top-level _1 fields from the given union type.
rmUnion_1 :: Type a -> Type a
rmUnion_1 typ = typ
  { extract = \expr -> extract typ (add1 expr)
  , expected = rm1 (expected typ)
  }
  where
    -- Add _1 to union expression
    add1 expr =
      case expr of
        Union m -> Union (fmap addField_1 m)
        UnionLit key val m -> UnionLit key (addField_1 val) (fmap addField_1 m)
        _ -> expr
    -- Remove _1 from union epxression
    rm1 expr =
      case expr of
        Union m -> Union (fmap rmField_1 m)
        UnionLit key val m -> UnionLit key (rmField_1 val) (fmap rmField_1 m)
        _ -> expr


-- | Add _1 in the given record expression.
addField_1 :: Expr s a -> Expr s a
addField_1 expr =
  case expr of
    RecordLit m -> RecordLit (Map.singleton "_1" (RecordLit m))
    Record m -> Record (Map.singleton "_1" (Record m))
    _ -> expr


-- | Remove _1 from the given record expression.
rmField_1 :: Expr s a -> Expr s a
rmField_1 expr =
  case expr of
    RecordLit m -> Prelude.maybe (RecordLit m) id $ do
      guard $ Map.keys m == ["_1"]
      RecordLit m' <- Map.lookup "_1" m
      return (RecordLit m')
    Record m -> Prelude.maybe (Record m) id $ do
      guard $ Map.keys m == ["_1"]
      Record m' <- Map.lookup "_1" m
      return (Record m')
    _ -> expr
