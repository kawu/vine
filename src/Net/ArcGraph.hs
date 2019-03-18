{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE OverloadedStrings #-}
-- {-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TupleSections #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- To derive Binary for `Param`
{-# LANGUAGE DeriveAnyClass #-}

-- -- To make GHC automatically infer that `KnownNat d => KnownNat (d + d)`
-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fconstraint-solver-iterations=5 #-}


module Net.ArcGraph
  ( 
  -- * Network
--     Config(..)
    Param
  -- , printParam
  -- , size
  , new
  , Graph(..)
  , Node(..)
  , Arc
  , run
  , eval
  , runQ
  , evalQ
  -- , runTest

  -- * Storage
  , saveParam
  , loadParam

  -- * Data set
  , DataSet
  , Elem(..)
  -- , mkGraph
  -- , mkSent

  -- * Error
  , netError
  , netErrorQ

  -- * Training
  -- , trainProg
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats

import           System.Random (randomRIO)

import           Control.Monad (forM_, forM)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
-- import qualified Data.List as L
import           Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Text as T
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as B
import           Codec.Compression.Zlib (compress, decompress)

-- import qualified Data.Map.Lens as ML
import           Control.Lens.At (ixAt)
import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)
import qualified Numeric.SGD.ParamSet as SGD

import           Net.ArcGraph.Graph
import           Net.ArcGraph.BiComp
import qualified Net.ArcGraph.QuadComp as Q

-- import qualified Embedding.Dict as D

import           Debug.Trace (trace)


----------------------------------------------
-- Parameters
----------------------------------------------


-- -- | Parameter set
-- type Param dim a b 
--    = Biaff dim dim
--   :& Bias


-- -- | Parameter set (ONE)
-- type Param dim a b 
--    = Biaff dim dim
--   :& UnordBiaff dim dim
--   :& HeadAff dim dim
--   :& DepAff dim dim
--   :& HeadPosAff a
--   :& DepPosAff a
--   :& PapyPosAff a
--   :& EnkelPosAff a
--   :& ArcBias b
--   :& PapyArcAff b
--   :& EnkelArcAff b
--   :& Bias


-- -- | Parameter set (HOLISTIC)
-- type Param dim a b 
--    = Holi dim 50 a b 100 100
-- --   :& UnordBiaff dim dim
-- --   :& HeadAff dim dim
-- --   :& DepAff dim dim
-- --   :& HeadPosAff a
-- --   :& DepPosAff a
-- --   :& PapyPosAff a
-- --   :& EnkelPosAff a
-- --   :& ArcBias b
-- --   :& PapyArcAff b
-- --   :& EnkelArcAff b
--   :& Bias


-- -- | Parameter set (TWO; ONE without unordered binary relation)
-- type Param dim a b
--    = Biaff dim dim
--   -- :& UnordBiaff dim dim
--   :& HeadAff dim dim
--   :& DepAff dim dim
--   :& HeadPosAff a
--   :& DepPosAff a
--   :& PapyPosAff a
--   :& EnkelPosAff a
--   :& ArcBias b
--   :& PapyArcAff b
--   :& EnkelArcAff b
--   :& Bias


-- -- | New, quad parameter set!
-- type Param d a b
-- --    = Q.QuadAff d d
--    = Q.TriAff d d
--   :& Q.SibAff d d
--   :& Q.BiAff d d
--   :& Q.UnAff d d
--   :& Q.Bias


-- | New, quad parameter set!
type Param d a b
   = Q.TriAff d 100
  :& Q.SibAff d 100
  :& Q.BiAff d 100
  :& Q.UnordBiAff d 100
  :& Q.UnAff d 100
  :& Q.Bias


-- printParam :: (Show a, Show b) => Param dim a b -> IO ()
-- printParam
--   ( biaff :& unordBiaff :& headAff :& depAff :&
--     headPosAff :& depPosAff :& papyPosAff :&
--     enkelPosAff :& arcBias :& papyArcAff :&
--     enkelArcAff :& bias
--   ) = do
--     print headPosAff
--     print depPosAff
--     print papyPosAff
--     print enkelPosAff
--     print arcBias
--     print papyArcAff
--     print enkelArcAff
--     print bias


run
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , BiComp dim a b comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = runBiComp graph arc net
  return (arc, logistic x)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , BiComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
eval net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        graph
    )


runQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , Q.QuadComp dim a b comp
     )
  => BVar s comp
    -- ^ Quad component
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
runQ net graph = normalize . M.unionsWith (+) $ do
  arc <- graphArcs graph
  return $ Q.runQuadComp graph arc net
  where
    normalize = fmap logistic


evalQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , Q.QuadComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
evalQ net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runQ
        (BP.constVar net)
        graph
    )


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam :: (Binary a) => FilePath -> a -> IO ()
saveParam path =
  B.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam :: (Binary a) => FilePath -> IO a
loadParam path =
  Bin.decode . decompress <$> B.readFile path
  -- B.writeFile path . compress . Bin.encode


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim a b = Elem
  { graph :: Graph (Node dim a) b
    -- ^ Input graph
  , labMap :: M.Map Arc Double
    -- ^ Target probabilities
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of (graph, target value map) pairs.
type DataSet d a b = [Elem d a b]


----------------------------------------------
-- Error
----------------------------------------------


-- -- | Squared error between the target and the actual output.
-- errorOne
--   :: (Ord a, KnownNat n, Reifies s W)
--   => M.Map a (BVar s (R n))
--     -- ^ Target values
--   -> M.Map a (BVar s (R n))
--     -- ^ Output values
--   -> BVar s Double
-- errorOne target output = PB.sum . BP.collectVar $ do
--   (key, tval) <- M.toList target
--   let oval = output M.! key
--       err = tval - oval
--   return $ err `dot` err


-- | Cross entropy between the true and the artificial distributions
crossEntropy
  :: (Reifies s W)
  => BVar s Double
    -- ^ Target ,,true'' MWE probability
  -> BVar s Double
    -- ^ Output ,,artificial'' MWE probability
  -> BVar s Double
crossEntropy p q = negate
  ( p0 * log q0
  + p1 * log q1
  )
  where
    p1 = p
    p0 = 1 - p1
    q1 = q
    q0 = 1 - q1


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: (KnownNat n, Reifies s W)
--   => BVar s (R n)
--     -- ^ Target ,,true'' distribution
--   -> BVar s (R n)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropy p q =
--   negate (p `dot` LBP.vmap' log q)


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
--   => M.Map a (BVar s (R n))
  => M.Map a (BVar s Double)
    -- ^ Target values
--   -> M.Map a (BVar s (R n))
  -> M.Map a (BVar s Double)
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
--   => [M.Map a (BVar s (R n))] -- ^ Targets
--   -> [M.Map a (BVar s (R n))] -- ^ Outputs
  => [M.Map a (BVar s Double)] -- ^ Targets
  -> [M.Map a (BVar s Double)] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: ( Reifies s W, KnownNat dim
     , Ord a, Show a, Ord b, Show b
     , BiComp dim a b comp
     )
  => DataSet dim a b
  -- -> BVar s (Param dim a b)
  -> BVar s comp
  -> BVar s Double
netError dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (run net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)


-- | Network error on a given dataset.
netErrorQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , Q.QuadComp dim a b comp
     )
  => DataSet dim a b
  -> BVar s comp
  -> BVar s Double
netErrorQ dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (runQ net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)
